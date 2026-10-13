import os
import copy

import numpy as np
import torch
import torch.nn as nn
from offline_moo.off_moo_baselines.data import spearman_correlation
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm
from typing import Union

from experiments.paretoflow_reference_directions import UniformReferenceDirectionFactory
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def empirical_pareto_front(Y: np.ndarray) -> np.ndarray:
    fronts = NonDominatedSorting().do(Y)
    return Y[fronts[0]]


def get_reference_directions(name, *args, **kwargs):
    REF = {
        "uniform": UniformReferenceDirectionFactory,
        "das-dennis": UniformReferenceDirectionFactory,
    }

    if name not in REF:
        raise Exception("Reference directions factory not found.")

    return REF[name](*args, **kwargs)()


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    """
    Cheap validation for flow matching.

    This replaces the previous NLL/log_prob evaluation with validation FM loss.

    API is unchanged:
        evaluation(test_loader, name=None, model_best=None, epoch=None)

    It computes the same objective used during training:
        loss = model_best(batch, reduction="mean")

    No log_prob.
    No divergence.
    No Hutchinson estimator.
    No sampling-based metric.
    """
    if model_best is None:
        model_best = torch.load(name + ".model")

    was_training = model_best.training
    model_best.eval()

    # Internal repeat count. No API/arg change.
    # The FM loss is stochastic because model.forward samples x_0 and t.
    # Repeating a few times gives a more stable validation estimate.
    fm_eval_repeats = 4

    total_loss = 0.0
    N = 0

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Validation FM", unit="batch") as pbar:
            for indx_batch, test_batch in enumerate(test_loader):
                test_batch = test_batch.float()
                test_batch = test_batch.to(device)

                batch_size = test_batch.shape[0]

                repeat_losses = []
                for _ in range(fm_eval_repeats):
                    loss_t = model_best(test_batch, reduction="mean")
                    repeat_losses.append(loss_t.item())

                batch_loss = float(np.mean(repeat_losses))

                total_loss += batch_loss * batch_size
                N += batch_size

                pbar.set_postfix({"fm_loss": total_loss / max(N, 1)})
                pbar.update(1)

    loss = total_loss / max(N, 1)

    if was_training:
        model_best.train()

    if epoch is None:
        print(f"FINAL LOSS: fm_val_loss={loss}")
    else:
        print(f"Epoch: {epoch}, val fm loss={loss}")

    return loss


def _set_requires_grad(model, requires_grad: bool):
    """
    Enable / disable gradients for all model parameters.

    Used for EMA model because EMA is only updated by moving average,
    not by backpropagation.
    """
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def _make_ema_model(model):
    """
    Create an EMA copy of the current model.

    The EMA model:
      - has the same architecture and initial weights;
      - is put in eval mode;
      - does not require parameter gradients.
    """
    import copy

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    _set_requires_grad(ema_model, False)

    return ema_model


@torch.no_grad()
def _ema_update(ema_model, model, decay: float = 0.999):
    """
    Update EMA parameters:

        theta_ema <- decay * theta_ema + (1 - decay) * theta

    Buffers are copied directly from the raw model.
    This is safer for fixed buffers such as time-embedding frequency buffers.
    """
    decay = float(decay)

    ema_params = dict(ema_model.named_parameters())
    raw_params = dict(model.named_parameters())

    for name, raw_param in raw_params.items():
        ema_param = ema_params[name]
        ema_param.data.mul_(decay).add_(raw_param.data, alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    raw_buffers = dict(model.named_buffers())

    for name, raw_buffer in raw_buffers.items():
        ema_buffers[name].copy_(raw_buffer)


def _get_ema_decay(global_step: int, target_decay: float = 0.999):
    """
    Adaptive EMA decay with warmup.

    Fixed decay=0.999 from step 1 can make EMA lag too much early in training.
    This rule starts with a smaller decay and gradually approaches target_decay.

    Example:
        early steps: EMA follows raw model more closely
        later steps: EMA becomes smoother and stable
    """
    step = float(global_step + 1)

    # Warmup rule. Starts around 0.18, then increases toward 1.
    warmup_decay = (1.0 + step) / (10.0 + step)

    return min(float(target_decay), float(warmup_decay))


def training(
    name, max_patience, num_epochs, model, optimizer, training_loader, val_loader
):
    """
    Train the flow-matching model with EMA.

    API unchanged.

    Current behavior:
      - train raw model every epoch;
      - update EMA after every optimizer step;
      - use adaptive EMA warmup instead of fixed EMA decay;
      - evaluate EMA model every epoch using cheap FM validation loss;
      - save EMA model every epoch to name + ".model";
      - save raw model every epoch to name + "_raw.model";
      - no early stopping for now.
    """
    fm_val = []

    # Kept for later early-stopping use, currently disabled.
    best_fm_val = float("inf")
    patience = 0

    # EMA target decay. Actual decay is warmed up by _get_ema_decay(...).
    target_ema_decay = 0.999
    global_step = 0

    ema_model = _make_ema_model(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    # Main loop
    for e in range(num_epochs):
        # ------------------------------------------------------------
        # Training raw model
        # ------------------------------------------------------------
        model.train()

        epoch_loss = 0.0
        last_ema_decay = 0.0

        with tqdm(
            total=len(training_loader),
            desc=f"Training {e + 1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for indx_batch, batch in enumerate(training_loader):
                batch = batch.float()
                batch = batch.to(device)

                loss = model(batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # EMA warmup update after every optimizer step.
                ema_decay = _get_ema_decay(
                    global_step=global_step,
                    target_decay=target_ema_decay,
                )
                last_ema_decay = ema_decay

                _ema_update(
                    ema_model=ema_model,
                    model=model,
                    decay=ema_decay,
                )

                global_step += 1

                epoch_loss = epoch_loss + loss.item()

                pbar.set_postfix(
                    {
                        "train_fm_loss": epoch_loss / (indx_batch + 1),
                        "ema_decay": last_ema_decay,
                    }
                )
                pbar.update(1)

        train_fm_loss = epoch_loss / len(training_loader)

        print(
            f"Epoch: {e}, "
            f"train fm loss={train_fm_loss}, "
            f"last ema decay={last_ema_decay}"
        )

        loss_val = evaluation(
            val_loader,
            model_best=ema_model,
            epoch=e,
        )

        fm_val.append(loss_val)

        # ------------------------------------------------------------
        # Save every epoch.
        #
        # Main checkpoint path stores EMA model.
        # Raw model is also saved for debugging / comparison.
        # ------------------------------------------------------------
        print("saved EMA model!")
        torch.save(ema_model, name + ".model")

        print("saved raw model!")
        torch.save(model, name + "_raw.model")

        # ------------------------------------------------------------
        # LR scheduler
        # ------------------------------------------------------------
        scheduler.step()

    fm_val = np.asarray(fm_val)

    return fm_val


class DesignDataset(Dataset):
    def __init__(self, designs):
        self.X = designs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# We don't use permutation tasks in our experiments
SyntheticFunctionDict = {
    "zdt1": "ZDT1-Exact-v0",
    "zdt2": "ZDT2-Exact-v0",
    "zdt3": "ZDT3-Exact-v0",
    "zdt4": "ZDT4-Exact-v0",
    "zdt6": "ZDT6-Exact-v0",
    "dtlz1": "DTLZ1-Exact-v0",
    "dtlz2": "DTLZ2-Exact-v0",
    "dtlz3": "DTLZ3-Exact-v0",
    "dtlz4": "DTLZ4-Exact-v0",
    "dtlz5": "DTLZ5-Exact-v0",
    "dtlz6": "DTLZ6-Exact-v0",
    "dtlz7": "DTLZ7-Exact-v0",
}

MONASSequenceDict = {
    "c10mop1": "C10MOP1-Exact-v0",
    "c10mop2": "C10MOP2-Exact-v0",
    "c10mop3": "C10MOP3-Exact-v0",
    "c10mop4": "C10MOP4-Exact-v0",
    "c10mop5": "C10MOP5-Exact-v0",
    "c10mop6": "C10MOP6-Exact-v0",
    "c10mop7": "C10MOP7-Exact-v0",
    "c10mop8": "C10MOP8-Exact-v0",
    "c10mop9": "C10MOP9-Exact-v0",
    "in1kmop1": "IN1KMOP1-Exact-v0",
    "in1kmop2": "IN1KMOP2-Exact-v0",
    "in1kmop3": "IN1KMOP3-Exact-v0",
    "in1kmop4": "IN1KMOP4-Exact-v0",
    "in1kmop5": "IN1KMOP5-Exact-v0",
    "in1kmop6": "IN1KMOP6-Exact-v0",
    "in1kmop7": "IN1KMOP7-Exact-v0",
    "in1kmop8": "IN1KMOP8-Exact-v0",
    "in1kmop9": "IN1KMOP9-Exact-v0",
}

MONASLogitsDict = {}

MOCOContinuousDict = {}

MORLDict = {
    "mo_swimmer_v2": "MOSwimmerV2-Exact-v0",
    "mo_hopper_v2": "MOHopperV2-Exact-v0",
}

ScientificDesignContinuousDict = {}

ScientificDesignSequenceDict = {}

RESuiteDict = {
    "re21": "RE21-Exact-v0",
    "re22": "RE22-Exact-v0",
    "re23": "RE23-Exact-v0",
    "re24": "RE24-Exact-v0",
    "re25": "RE25-Exact-v0",
    "re31": "RE31-Exact-v0",
    "re32": "RE32-Exact-v0",
    "re33": "RE33-Exact-v0",
    "re34": "RE34-Exact-v0",
    "re35": "RE35-Exact-v0",
    "re36": "RE36-Exact-v0",
    "re37": "RE37-Exact-v0",
    "re41": "RE41-Exact-v0",
    "re42": "RE42-Exact-v0",
    "re61": "RE61-Exact-v0",
}

SyntheticFunction = list(SyntheticFunctionDict.values())
MONASSequence = list(MONASSequenceDict.values())
MONASLogits = list(MONASLogitsDict.values())
MOCOContinuous = list(MOCOContinuousDict.values())
MORL = list(MORLDict.values())
ScientificDesignContinuous = list(ScientificDesignContinuousDict.values())
ScientificDesignSequence = list(ScientificDesignSequenceDict.values())
RESuite = list(RESuiteDict.values())

MONAS = MONASSequence + MONASLogits
MOCO = MOCOContinuous
ScientificDesign = ScientificDesignContinuous + ScientificDesignSequence

ALLTASKS = SyntheticFunction + MONAS + MOCO + MORL + ScientificDesign + RESuite
ALLTASKSDICT = {
    **SyntheticFunctionDict,
    **MONASSequenceDict,
    **MONASLogitsDict,
    **MOCOContinuousDict,
    **MORLDict,
    **ScientificDesignContinuousDict,
    **ScientificDesignSequenceDict,
    **RESuiteDict,
}

CONTINUOUSTASKS = (
    SyntheticFunction
    + MONASLogits
    + MOCOContinuous
    + MORL
    + ScientificDesignContinuous
    + RESuite
)
SEQUENCETASKS = MONASSequence + ScientificDesignSequence

# Get all keys in the dictionary
all_task_names = list(ALLTASKSDICT.keys())


class SingleModelBaseTrainer(nn.Module):

    def __init__(self, model, which_obj, args):
        super(SingleModelBaseTrainer, self).__init__()
        self.args = args

        # ------------------------------------------------------------
        # Keep the original public-arg behavior.
        # Use bash / existing args to set:
        #   --proxies_lr
        #   --proxies_lr_decay
        #   --proxies_epochs
        # ------------------------------------------------------------
        self.forward_lr = args.proxies_lr
        self.forward_lr_decay = args.proxies_lr_decay
        self.n_epochs = args.proxies_epochs

        self.model = model
        self.which_obj = which_obj

        # ------------------------------------------------------------
        # Revert to original optimizer and original loss.
        # ------------------------------------------------------------
        self.forward_opt = Adam(model.parameters(), lr=args.proxies_lr)

        self.train_criterion = lambda yhat, y: (
            torch.sum(torch.mean((yhat - y) ** 2, dim=1))
        )

        self.mse_criterion = nn.MSELoss()

        # ------------------------------------------------------------
        # Internal early stopping configuration.
        # No new args / APIs.
        #
        # Patience 40 is in the requested 30-50 range.
        # Early stopping and checkpointing are both based on validation MSE.
        # ------------------------------------------------------------
        self.early_stopping_patience = 50
        self.early_stopping_min_delta = 1e-8

        self.best_epoch = -1
        self.best_val_mse = float("inf")
        self.best_val_pcc = -float("inf")

    def _is_better_val_mse(self, val_mse_value):
        """
        Check whether current validation MSE is meaningfully better.

        Validation MSE is the early-stopping metric and the checkpoint
        selection metric.
        """
        if self.best_epoch < 0:
            return True

        return val_mse_value < self.best_val_mse - self.early_stopping_min_delta

    def _maybe_save_checkpoint(self, epoch, val_mse, val_pcc):
        """
        Save checkpoint if validation MSE improves.

        Important:
        - Checkpoint selection is now based on validation MSE.
        - We still save valid_pcc because SingleModel.load() expects it.
        """
        val_mse_value = float(val_mse.detach().cpu().item())
        val_pcc_value = float(val_pcc.detach().cpu().item())

        if self._is_better_val_mse(val_mse_value):
            print(
                "🌸 New best epoch by validation MSE! "
                f"epoch={epoch + 1}, "
                f"val_mse={val_mse_value:.6e}, "
                f"val_pcc={val_pcc_value:.6f}"
            )

            self.best_epoch = int(epoch)
            self.best_val_mse = val_mse_value
            self.best_val_pcc = val_pcc_value

            # Keep old checkpoint format compatible:
            # {
            #   "model_state_dict": ...,
            #   "valid_pcc": ...
            # }
            self.model.save(val_pcc=val_pcc_value)

            return True

        return False

    def _evaluate_performance(self, statistics, epoch, train_loader, val_loader):
        self.model.eval()

        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                y_all = torch.cat((y_all, batch_y), dim=0)

                outputs = self.model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

            train_mse = self.mse_criterion(outputs_all, y_all)
            train_corr = spearman_correlation(outputs_all, y_all)
            train_pcc = self.compute_pcc(outputs_all, y_all)

            statistics[f"model_{self.which_obj}/train/mse"] = train_mse.item()

            for i in range(self.n_obj):
                statistics[f"model_{self.which_obj}/train/rank_corr_{i + 1}"] = (
                    train_corr[i].item()
                )

            print(
                "Epoch [{}/{}], MSE: {:}, PCC: {:}".format(
                    epoch + 1,
                    self.n_epochs,
                    train_mse.item(),
                    train_pcc.item(),
                )
            )

        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)

            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                y_all = torch.cat((y_all, batch_y), dim=0)

                outputs = self.model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

            val_mse = self.mse_criterion(outputs_all, y_all)
            val_corr = spearman_correlation(outputs_all, y_all)
            val_pcc = self.compute_pcc(outputs_all, y_all)

            statistics[f"model_{self.which_obj}/valid/mse"] = val_mse.item()

            for i in range(self.n_obj):
                statistics[f"model_{self.which_obj}/valid/rank_corr_{i + 1}"] = (
                    val_corr[i].item()
                )

            print(
                "Valid MSE: {:}, Valid PCC: {:}".format(
                    val_mse.item(),
                    val_pcc.item(),
                )
            )

            improved = self._maybe_save_checkpoint(
                epoch=epoch,
                val_mse=val_mse,
                val_pcc=val_pcc,
            )

        return statistics, improved

    def launch(
        self,
        train_loader=None,
        val_loader=None,
        retrain_model: bool = True,
    ):

        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if not retrain_model and os.path.exists(self.model.save_path):
            self.model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None

        # Reset best-checkpoint state at the beginning of training.
        self.best_epoch = -1
        self.best_val_mse = float("inf")
        self.best_val_pcc = -float("inf")

        early_stopping_counter = 0
        statistics = {}

        for epoch in range(self.n_epochs):
            self.model.train()

            losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                self.forward_opt.zero_grad()

                outputs = self.model(batch_x)
                loss = self.train_criterion(outputs, batch_y)

                losses.append(loss.item() / batch_x.size(0))

                loss.backward()
                self.forward_opt.step()

            statistics[f"model_{self.which_obj}/train/loss/mean"] = np.array(
                losses
            ).mean()
            statistics[f"model_{self.which_obj}/train/loss/std"] = np.array(
                losses
            ).std()
            statistics[f"model_{self.which_obj}/train/loss/max"] = np.array(
                losses
            ).max()

            statistics, improved = self._evaluate_performance(
                statistics,
                epoch,
                train_loader,
                val_loader,
            )

            if improved:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            print(
                f"[EarlyStopping] counter={early_stopping_counter}/"
                f"{self.early_stopping_patience}, "
                f"best_epoch={self.best_epoch + 1}, "
                f"best_val_mse={self.best_val_mse:.6e}, "
                f"best_val_pcc={self.best_val_pcc:.6f}"
            )

            if early_stopping_counter >= self.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch={self.best_epoch + 1}, "
                    f"best_val_mse={self.best_val_mse:.6e}, "
                    f"best_val_pcc={self.best_val_pcc:.6f}"
                )
                break

            statistics[f"model_{self.which_obj}/train/lr"] = self.forward_lr
            self.forward_lr *= self.forward_lr_decay
            update_lr(self.forward_opt, self.forward_lr)

    def compute_pcc(self, valid_preds, valid_labels):
        vx = valid_preds - torch.mean(valid_preds)
        vy = valid_labels - torch.mean(valid_labels)

        pcc = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2) + 1e-12) * torch.sqrt(torch.sum(vy**2) + 1e-12)
        )

        return pcc


class MultipleModels(nn.Module):
    def __init__(
        self, n_dim, n_obj, hidden_size, train_mode, save_dir=None, save_prefix=None
    ):
        super(MultipleModels, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        self.obj2model = {}
        self.hidden_size = hidden_size
        self.train_mode = train_mode

        self.save_dir = save_dir
        self.save_prefix = save_prefix
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        for obj in range(self.n_obj):
            self.create_models(obj)

    def create_models(self, learning_obj):
        model = SingleModel
        new_model = model(
            self.n_dim,
            self.hidden_size,
            which_obj=learning_obj,
            save_dir=self.save_dir,
            save_prefix=self.save_prefix,
        )
        self.obj2model[learning_obj] = new_model

    def set_kwargs(self, device=None, dtype=None):
        for model in self.obj2model.values():
            model.set_kwargs(device=device, dtype=dtype)
            model.to(device=device, dtype=dtype)

    def forward(self, x, forward_objs=None):
        if forward_objs is None:
            forward_objs = list(self.obj2model.keys())
        x = [self.obj2model[obj](x) for obj in forward_objs]
        x = torch.cat(x, dim=1)
        return x


activate_functions = [nn.LeakyReLU(), nn.LeakyReLU()]


class SingleModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, which_obj, save_dir=None, save_prefix=None
    ):
        super(SingleModel, self).__init__()
        self.n_dim = input_size
        self.n_obj = 1
        self.which_obj = which_obj
        self.activate_functions = activate_functions

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size) - 1], 1))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size

        self.save_path = os.path.join(save_dir, f"{save_prefix}-{which_obj}.pt")

    def forward(self, x):
        for i in range(len(self.hidden_size)):
            x = self.layers[i](x)
            x = self.activate_functions[i](x)

        x = self.layers[len(self.hidden_size)](x)
        out = x

        return out

    def set_kwargs(self, device=None, dtype=None):
        self.to(device=device, dtype=dtype)

    def check_model_path_exist(self, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        return os.path.exists(save_path)

    def save(self, val_pcc=None, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path

        self = self.to("cpu")
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if val_pcc is not None:
            checkpoint["valid_pcc"] = val_pcc

        torch.save(checkpoint, save_path)
        self = self.to(**tkwargs)

    def load(self, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path

        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        valid_pcc = checkpoint["valid_pcc"]
        print(
            f"Successfully load trained model from {save_path} "
            f"with valid PCC = {valid_pcc}"
        )


def get_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.9,
    batch_size: int = 32,
):

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(**tkwargs)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).to(**tkwargs)

    tensor_dataset = TensorDataset(X, y)
    lengths = [
        int(val_ratio * len(tensor_dataset)),
        len(tensor_dataset) - int(val_ratio * len(tensor_dataset)),
    ]
    train_dataset, val_dataset = random_split(tensor_dataset, lengths)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 4, shuffle=False, drop_last=False
    )

    return train_loader, val_loader


@torch.no_grad()
def out_of_bounds_ratio_in_normalized_space(
    X_norm: Union[np.ndarray, torch.Tensor],
    xl=None,
    xu=None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> float:
    """
    Ratio of samples outside [xl, xu] in normalized space, with loose tolerance.
    Works for X_norm as numpy or torch. Returns float in [0,1].
    """
    if xl is None or xu is None:
        return 0.0

    if isinstance(X_norm, torch.Tensor):
        X = X_norm.detach().cpu().numpy()
    else:
        X = np.asarray(X_norm)

    xl = np.asarray(xl).reshape(1, -1)
    xu = np.asarray(xu).reshape(1, -1)

    # Ensure shape compatible (continuous case: (N, d))
    Xf = X.reshape(X.shape[0], -1)
    if Xf.shape[1] != xl.shape[1] or xl.shape != xu.shape:
        raise ValueError(
            f"Shape mismatch: X={X.shape}, Xf={Xf.shape}, xl={xl.shape}, xu={xu.shape}"
        )

    finite = np.isfinite(Xf).all(axis=1)

    tol_lo = atol + rtol * np.abs(xl)
    tol_hi = atol + rtol * np.abs(xu)

    below = Xf < (xl - tol_lo)
    above = Xf > (xu + tol_hi)

    viol_any = (below | above).any(axis=1)
    out = (~finite) | viol_any
    return float(out.mean())
