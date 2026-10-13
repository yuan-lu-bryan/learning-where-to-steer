from typing import List

import numpy as np

from off_moo_bench.datasets.dataset_builder import DatasetBuilder


def one_hot(a, num_classes):
    """A helper function that converts integers into a floating
    point one-hot representation using pure numpy:
    https://stackoverflow.com/questions/36960320/
    convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy

    """

    out = np.zeros((a.size, num_classes), dtype=np.float32)
    out[np.arange(a.size), a.ravel()] = 1.0
    out.shape = a.shape + (num_classes,)
    return out


class DiscreteDataset(DatasetBuilder):
    name = "DiscreteDataset"
    x_name = "Design"
    y_name = "Prediction"

    @property
    def subclass_kwargs(self):
        return dict(
            is_logits=self.is_logits,
            soft_interpolation=self.soft_interpolation,
            num_classes_on_each_position=self.num_classes_on_each_position,
            class_lower_bounds=self.class_lower_bounds,
            class_upper_bounds=self.class_upper_bounds,
        )

    @property
    def subclass(self):
        return DiscreteDataset

    def __init__(
        self,
        *args,
        is_logits=False,
        num_classes_on_each_position: List[int] = None,
        class_lower_bounds: List[int] = None,
        class_upper_bounds: List[int] = None,
        soft_interpolation=0.6,
        **kwargs,
    ):
        self.soft_interpolation = soft_interpolation
        self.num_classes_on_each_position = (
            list(num_classes_on_each_position)
            if num_classes_on_each_position is not None
            else None
        )
        self.class_lower_bounds = (
            list(class_lower_bounds) if class_lower_bounds is not None else None
        )
        self.class_upper_bounds = (
            list(class_upper_bounds) if class_upper_bounds is not None else None
        )
        self.is_logits = is_logits

        # initialize the dataset using the method in the base class
        super(DiscreteDataset, self).__init__(*args, **kwargs)

    def batch_transform(self, x_batch, y_batch, return_x=True, return_y=True):
        # convert the design values from integers to logits
        if self.is_logits and return_x and np.issubdtype(x_batch.dtype, np.integer):
            x_batch = self.to_logits(x_batch)

        # convert the design values from logits to integers
        elif (
            not self.is_logits
            and return_x
            and np.issubdtype(x_batch.dtype, np.floating)
        ):
            x_batch = self.to_integers(x_batch)

        return super(DiscreteDataset, self).batch_transform(
            x_batch, y_batch, return_x=return_x, return_y=return_y
        )

    def update_x_statistics(self):
        # handle corner case when we need statistics but they were
        # not computed yet and the dataset is currently mapped to integers
        original_is_logits = self.is_logits
        self.is_logits = True
        super(DiscreteDataset, self).update_x_statistics()
        self.is_logits = original_is_logits

    def map_normalize_x(self):
        # check that the dataset is in a form that supports normalization
        if not self.is_logits:
            raise ValueError("cannot normalize discrete design values")

        # call the normalization method of the super class
        super(DiscreteDataset, self).map_normalize_x()

    def normalize_x(self, x):
        # check that the dataset is in a form that supports normalization
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot normalize discrete design values")

        # call the normalization method of the super class
        return super(DiscreteDataset, self).normalize_x(x)

    def map_denormalize_x(self):
        # check that the dataset is in a form that supports denormalization
        if not self.is_logits:
            raise ValueError("cannot denormalize discrete design values")

        # call the normalization method of the super class
        super(DiscreteDataset, self).map_denormalize_x()

    def denormalize_x(self, x):
        # check that the dataset is in a form that supports denormalization
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot denormalize discrete design values")

        # call the normalization method of the super class
        return super(DiscreteDataset, self).denormalize_x(x)

    def help_to_logits(
        self, x: np.ndarray, num_classes: int, soft_interpolation: float = 0.6
    ):
        """
        Convert the integers to logits.
        :param x: np.ndarray: the input data, with shape (n_samples, 1)
        :param num_classes: int: the number of classes
        :param soft_interpolation: float: the soft interpolation parameter
        :return: np.ndarray: the logits, with shape (n_samples, 1, n_classes)
        """
        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("cannot convert non-integers to logits")

        # convert the integers to one hot vectors
        one_hot_x = one_hot(x, num_classes)

        # build a uniform distribution to interpolate between
        uniform_prior = np.full_like(one_hot_x, 1 / float(num_classes))

        # interpolate between a dirac distribution and a uniform prior
        soft_x = (
            soft_interpolation * one_hot_x + (1.0 - soft_interpolation) * uniform_prior
        )

        # convert to log probabilities
        x = np.log(soft_x)

        # remove one degree of freedom caused by \sum_i p_i = 1.0
        return (x[:, :, 1:] - x[:, :, :1]).astype(np.float32)

    def to_logits(
        self,
        x: np.ndarray,
        soft_interpolation: float = 0.6,
    ):
        self._check_discrete_metadata()

        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("cannot convert non-integers to logits")

        effective_num_classes = self._effective_num_classes()
        logits = []
        sequence_length = len(effective_num_classes)

        for i in range(sequence_length):
            K_true = int(self.num_classes_on_each_position[i])
            K_eff = int(effective_num_classes[i])
            lb = int(self.class_lower_bounds[i])

            temp_x = x[:, i].reshape(-1, 1).astype(np.int32)

            # map raw MONAS label -> local zero-based class index
            local_x = temp_x - lb

            if np.any(local_x < 0) or np.any(local_x >= K_true):
                raise ValueError(
                    f"position {i}: raw values out of bounds, "
                    f"expected [{lb}, {lb + K_true - 1}]"
                )

            logits.append(self.help_to_logits(local_x, K_eff, soft_interpolation))

        return np.concatenate(logits, axis=2).squeeze()

    def help_to_integers(self, x: np.ndarray, true_num_of_classes: int):
        """
        Convert the logits to integers.
        :param x: np.ndarray: the input data, with shape (n_samples, 1, n_classes)
        :param true_num_of_classes: int: the true number of classes
        :return: np.ndarray: the integers, with shape (n_samples, 1)
        """
        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot convert non-floats to integers")

        # Since we might add a dummy class, we need to make sure that the last class is not selected
        # For RFP-Exact-v0
        if true_num_of_classes == 1:
            return np.zeros(x.shape[:-1], dtype=np.int32)

        res = np.argmax(
            np.pad(x, [[0, 0]] * (len(x.shape) - 1) + [[1, 0]]), axis=-1
        ).astype(np.int32)
        return res

    def to_integers(self, x: np.ndarray):
        self._check_discrete_metadata()

        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot convert non-floats to integers")

        effective_num_classes = self._effective_num_classes()
        integers = []
        start = 0
        sequence_length = len(effective_num_classes)

        for i in range(sequence_length):
            K_true = int(self.num_classes_on_each_position[i])
            K_eff = int(effective_num_classes[i])
            lb = int(self.class_lower_bounds[i])

            width = K_eff - 1
            temp_x = x[:, start : (start + width)].reshape(-1, 1, width)

            local_idx = self.help_to_integers(temp_x, K_true)
            raw_idx = local_idx + lb

            integers.append(raw_idx.astype(np.int32))
            start += width

        return np.concatenate(integers, axis=1)

    def _check_discrete_metadata(self):
        if self.num_classes_on_each_position is None:
            raise ValueError("num_classes_on_each_position is not set")
        if self.class_lower_bounds is None:
            raise ValueError("class_lower_bounds is not set")
        if len(self.num_classes_on_each_position) != len(self.class_lower_bounds):
            raise ValueError(
                "num_classes_on_each_position and class_lower_bounds length mismatch"
            )
        if self.class_upper_bounds is not None and len(self.class_upper_bounds) != len(
            self.class_lower_bounds
        ):
            raise ValueError("class_upper_bounds length mismatch")

    def _effective_num_classes(self):
        self._check_discrete_metadata()
        return [2 if k == 1 else int(k) for k in self.num_classes_on_each_position]

    def map_to_logits(self):
        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if not self.is_logits:
            # set the appropriate state variable
            self.is_logits = True

            # check shape and data type of a single design value x
            for x0 in self.iterate_samples(return_y=False):
                self.input_shape = x0.shape
                self.input_size = int(np.prod(x0.shape))
                self.input_dtype = x0.dtype
                break

            for x0 in self.iterate_test_samples(return_y=False):
                assert self.input_shape == x0.shape
                assert self.input_size == int(np.prod(x0.shape))
                assert self.input_dtype == x0.dtype
                break

    def map_to_integers(self):
        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if self.is_logits:
            if self.is_normalized_x:
                self.map_denormalize_x()

            # set the appropriate state variable
            self.is_logits = False

            # check shape and data type of a single design value x
            for x0 in self.iterate_samples(return_y=False):
                self.input_shape = x0.shape
                self.input_size = int(np.prod(x0.shape))
                self.input_dtype = x0.dtype
                break

            for x0 in self.iterate_test_samples(return_y=False):
                assert self.input_shape == x0.shape
                assert self.input_size == int(np.prod(x0.shape))
                assert self.input_dtype == x0.dtype
                break
