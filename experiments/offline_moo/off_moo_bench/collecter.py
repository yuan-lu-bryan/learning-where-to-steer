"""Genetic operators used by the paper's surrogate baselines."""

import numpy as np
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import FloatRandomSampling

try:
    from off_moo_bench.problem.mo_nas import get_genetic_operator

    evox_sampling, evox_crossover, evox_mutation, evox_repair = get_genetic_operator()
except Exception:
    evox_sampling = evox_crossover = evox_mutation = evox_repair = None


class RoundingRepair(Repair):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        return np.around(X).astype(int)


class IntegerRandomSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        return np.around(X).astype(int)


CROSSOVERS = {"evox_crossover": evox_crossover}
MUTATIONS = {"evox_mutation": evox_mutation}
REPAIRS = {"evox_repair": evox_repair, "rounding": RoundingRepair()}
SAMPLINGS = {"evox_sampling": evox_sampling, "int_rnd": IntegerRandomSampling()}


def get_operator_dict(config: dict) -> dict:
    operator_dict = {}
    if "crossover" in config.keys():
        assert (
            config["crossover"] in CROSSOVERS.keys()
        ), "Crossover {crossover} not found".format(crossover=config["crossover"])
        operator_dict["crossover"] = CROSSOVERS[config["crossover"]]

    if "mutation" in config.keys():
        assert (
            config["mutation"] in MUTATIONS.keys()
        ), "Mutation {mutation} not found".format(mutation=config["mutation"])
        operator_dict["mutation"] = MUTATIONS[config["mutation"]]

    if "repair" in config.keys():
        assert config["repair"] in REPAIRS.keys(), "Repair {repair} not found".format(
            repair=config["repair"]
        )
        operator_dict["repair"] = REPAIRS[config["repair"]]

    if "sampling" in config.keys():
        assert (
            config["sampling"] in SAMPLINGS.keys()
        ), "Sampling {sampling} not found".format(sampling=config["sampling"])
        operator_dict["sampling"] = SAMPLINGS[config["sampling"]]

    return operator_dict
