"""Registrations for the 47 tasks evaluated in the paper."""

try:
    # Config EvoXBench
    import os

    from evoxbench.database.init import config

    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "problem", "mo_nas"
    )
    config(os.path.join(base_path, "database"), os.path.join(base_path, "data"))
except:
    pass

import numpy as np

from off_moo_bench.registration import make, register, registry, spec

try:
    register(
        "RE21-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE21Dataset",
        "off_moo_bench.problem.synthetic_func:RE21",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE22-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE22Dataset",
        "off_moo_bench.problem.synthetic_func:RE22",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE23-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE23Dataset",
        "off_moo_bench.problem.synthetic_func:RE23",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE24-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE24Dataset",
        "off_moo_bench.problem.synthetic_func:RE24",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE25-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE25Dataset",
        "off_moo_bench.problem.synthetic_func:RE25",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE31-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE31Dataset",
        "off_moo_bench.problem.synthetic_func:RE31",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE32-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE32Dataset",
        "off_moo_bench.problem.synthetic_func:RE32",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE33-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE33Dataset",
        "off_moo_bench.problem.synthetic_func:RE33",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE34-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE34Dataset",
        "off_moo_bench.problem.synthetic_func:RE34",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE35-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE35Dataset",
        "off_moo_bench.problem.synthetic_func:RE35",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE36-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE36Dataset",
        "off_moo_bench.problem.synthetic_func:RE36",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE37-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE37Dataset",
        "off_moo_bench.problem.synthetic_func:RE37",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE41-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE41Dataset",
        "off_moo_bench.problem.synthetic_func:RE41",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE42-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE42Dataset",
        "off_moo_bench.problem.synthetic_func:RE42",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "RE61-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE61Dataset",
        "off_moo_bench.problem.synthetic_func:RE61",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "ZDT1-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT1Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT1",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "ZDT2-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT2Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "ZDT3-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT3Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT3",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "ZDT4-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT4Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT4",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "ZDT6-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT6Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT6",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "DTLZ1-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ1Dataset",
        "off_moo_bench.problem.dtlz:DTLZ1",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "DTLZ2-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ2Dataset",
        "off_moo_bench.problem.dtlz:DTLZ2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "DTLZ3-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ3Dataset",
        "off_moo_bench.problem.dtlz:DTLZ3",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "DTLZ4-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ4Dataset",
        "off_moo_bench.problem.dtlz:DTLZ4",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "DTLZ5-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ5Dataset",
        "off_moo_bench.problem.dtlz:DTLZ5",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "DTLZ6-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ6Dataset",
        "off_moo_bench.problem.dtlz:DTLZ6",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "DTLZ7-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ7Dataset",
        "off_moo_bench.problem.dtlz:DTLZ7",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "MOHopperV2-Exact-v0",
        "off_moo_bench.datasets.continuous.mo_hopper_dataset:MOHopperV2Dataset",
        "off_moo_bench.problem.morl.morl_problem:MOHopperV2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "MOSwimmerV2-Exact-v0",
        "off_moo_bench.datasets.continuous.mo_swimmer_dataset:MOSwimmerV2Dataset",
        "off_moo_bench.problem.morl.morl_problem:MOSwimmerV2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "C10MOP1-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:C10MOP1Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP1",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "C10MOP2-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:C10MOP2Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "C10MOP3-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:C10MOP3Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP3",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[8, 8, 8, 8, 8],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "C10MOP4-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:C10MOP4Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP4",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[8, 8, 8, 8, 8],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "C10MOP5-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:C10MOP5Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP5",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[5, 5, 5, 5, 5, 5],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "C10MOP6-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:C10MOP6Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP6",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[5, 5, 5, 5, 5, 5],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "C10MOP7-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:C10MOP7Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP7",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[5, 5, 5, 5, 5, 5],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "C10MOP8-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:C10MOP8Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP8",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                7,
                2,
                7,
                2,
                7,
                3,
                7,
                3,
                7,
                4,
                7,
                4,
                7,
                5,
                7,
                5,
                7,
                2,
                7,
                2,
                7,
                3,
                7,
                3,
                7,
                4,
                7,
                4,
                7,
                5,
                7,
                5,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "C10MOP9-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:C10MOP9Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP9",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                7,
                2,
                7,
                2,
                7,
                3,
                7,
                3,
                7,
                4,
                7,
                4,
                7,
                5,
                7,
                5,
                7,
                2,
                7,
                2,
                7,
                3,
                7,
                3,
                7,
                4,
                7,
                4,
                7,
                5,
                7,
                5,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "IN1KMOP1-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:IN1KMOP1Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP1",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                8,
                4,
                4,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                3,
                3,
                4,
                4,
                3,
                3,
                3,
                3,
                4,
                4,
                3,
                3,
                4,
                4,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "IN1KMOP2-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:IN1KMOP2Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                8,
                4,
                4,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                3,
                3,
                4,
                4,
                3,
                3,
                3,
                3,
                4,
                4,
                3,
                3,
                4,
                4,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "IN1KMOP3-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:IN1KMOP3Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP3",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                8,
                4,
                4,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                3,
                3,
                4,
                4,
                3,
                3,
                3,
                3,
                4,
                4,
                3,
                3,
                4,
                4,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "IN1KMOP4-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:IN1KMOP4Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP4",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "IN1KMOP5-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:IN1KMOP5Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP5",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "IN1KMOP6-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:IN1KMOP6Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP6",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "IN1KMOP7-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:IN1KMOP7Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP7",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                5,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "IN1KMOP8-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:IN1KMOP8Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP8",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                5,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass

try:
    register(
        "IN1KMOP9-Exact-v0",
        "off_moo_bench.datasets.discrete.monas_dataset:IN1KMOP9Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP9",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(
            max_samples=None,
            max_percentile=100,
            min_percentile=0,
            num_classes_on_each_position=[
                5,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
                9,
                9,
                10,
                10,
            ],
        ),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass
