from typing import List

import numpy as np
import torch
from pymoo.util.misc import find_duplicates
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


def calc_crowding_distance(F) -> np.ndarray:
    if isinstance(F, list) or isinstance(F, np.ndarray):
        F = torch.tensor(F).to(**tkwargs)

    n_points, n_obj = F.shape

    # sort each column and get index
    I = torch.argsort(F, dim=0, descending=False)

    # sort the objective space values for the whole matrix
    F_sorted = torch.gather(F, 0, I)

    # calculate the distance from each point to the last and next
    inf_tensor = torch.full((1, n_obj), float("inf"), device=F.device, dtype=F.dtype)
    neg_inf_tensor = torch.full(
        (1, n_obj), float("-inf"), device=F.device, dtype=F.dtype
    )
    dist = torch.cat([F_sorted, inf_tensor], dim=0) - torch.cat(
        [neg_inf_tensor, F_sorted], dim=0
    )

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = torch.max(F_sorted, dim=0).values - torch.min(F_sorted, dim=0).values
    norm[norm == 0] = float("nan")

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist[:-1], dist[1:]
    dist_to_last, dist_to_next = dist_to_last / norm, dist_to_next / norm

    dist_to_last[torch.isnan(dist_to_last)] = 0.0
    dist_to_next[torch.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = torch.argsort(I, dim=0, descending=False)
    crowding_dist = (
        torch.sum(
            torch.gather(dist_to_last, 0, J) + torch.gather(dist_to_next, 0, J), dim=1
        )
        / n_obj
    )

    return crowding_dist.detach().cpu().numpy()


def get_N_nondominated_indices(Y, num_ret, fronts=None) -> List[int]:
    """
    Select up to num_ret indices by nondominated sorting.

    Assumptions:
        - Objectives are minimization objectives.
        - Crowding distance is used only to truncate the last partially
          selected front.

    Important:
        - Returned indices are global indices into Y.
        - If fronts is provided, it should be the output of
          NonDominatedSorting().do(Y).
    """
    assert num_ret >= 0

    if num_ret == 0:
        return []

    # Convert Y to a NumPy array for pymoo utilities.
    # This makes the function robust to both NumPy arrays and torch tensors.
    if isinstance(Y, torch.Tensor):
        Y_np = Y.detach().cpu().numpy()
    else:
        Y_np = np.asarray(Y)

    if len(Y_np) == 0:
        return []

    num_ret = min(int(num_ret), len(Y_np))

    if fronts is None:
        fronts = NonDominatedSorting().do(
            Y_np,
            n_stop_if_ranked=num_ret,
        )

    indices_cnt = 0
    indices_select = []

    for front in fronts:
        front = np.asarray(front, dtype=int)

        if len(front) == 0:
            continue

        # If this whole front fits, keep the whole front.
        if indices_cnt + len(front) <= num_ret:
            indices_select += [int(i) for i in front]
            indices_cnt += len(front)

            if indices_cnt >= num_ret:
                break

        # Otherwise, truncate this front by crowding distance.
        else:
            n_keep = num_ret - indices_cnt

            if n_keep <= 0:
                break

            F = Y_np[front]

            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]

            d = np.zeros(len(front), dtype=float)

            if len(is_unique) > 0:
                _F = F[is_unique]
                _d = calc_crowding_distance(_F)
                d[is_unique] = _d

            # I_local indexes positions inside `front`, not global Y indices.
            I_local = np.argsort(d, kind="mergesort")[-n_keep:]

            # Critical fix: map local front positions back to global indices.
            I_global = front[I_local]

            indices_select += [int(i) for i in I_global]
            break

    return indices_select
