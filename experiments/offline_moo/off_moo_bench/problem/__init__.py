"""Load only the requested paper task and its optional oracle dependencies."""

from importlib import import_module

_PROBLEMS = {
    "re21": ("synthetic_func", "RE21"),
    "re22": ("synthetic_func", "RE22"),
    "re23": ("synthetic_func", "RE23"),
    "re24": ("synthetic_func", "RE24"),
    "re25": ("synthetic_func", "RE25"),
    "re31": ("synthetic_func", "RE31"),
    "re32": ("synthetic_func", "RE32"),
    "re33": ("synthetic_func", "RE33"),
    "re34": ("synthetic_func", "RE34"),
    "re35": ("synthetic_func", "RE35"),
    "re36": ("synthetic_func", "RE36"),
    "re37": ("synthetic_func", "RE37"),
    "re41": ("synthetic_func", "RE41"),
    "re42": ("synthetic_func", "RE42"),
    "re61": ("synthetic_func", "RE61"),
    "dtlz1": ("dtlz", "DTLZ1"),
    "dtlz2": ("dtlz", "DTLZ2"),
    "dtlz3": ("dtlz", "DTLZ3"),
    "dtlz4": ("dtlz", "DTLZ4"),
    "dtlz5": ("dtlz", "DTLZ5"),
    "dtlz6": ("dtlz", "DTLZ6"),
    "dtlz7": ("dtlz", "DTLZ7"),
    "zdt1": ("synthetic_func", "ZDT1"),
    "zdt2": ("synthetic_func", "ZDT2"),
    "zdt3": ("synthetic_func", "ZDT3"),
    "zdt4": ("synthetic_func", "ZDT4"),
    "zdt6": ("synthetic_func", "ZDT6"),
    "c10mop1": ("mo_nas.mo_nas", "C10MOP1"),
    "c10mop2": ("mo_nas.mo_nas", "C10MOP2"),
    "c10mop3": ("mo_nas.mo_nas", "C10MOP3"),
    "c10mop4": ("mo_nas.mo_nas", "C10MOP4"),
    "c10mop5": ("mo_nas.mo_nas", "C10MOP5"),
    "c10mop6": ("mo_nas.mo_nas", "C10MOP6"),
    "c10mop7": ("mo_nas.mo_nas", "C10MOP7"),
    "c10mop8": ("mo_nas.mo_nas", "C10MOP8"),
    "c10mop9": ("mo_nas.mo_nas", "C10MOP9"),
    "in1kmop1": ("mo_nas.mo_nas", "IN1KMOP1"),
    "in1kmop2": ("mo_nas.mo_nas", "IN1KMOP2"),
    "in1kmop3": ("mo_nas.mo_nas", "IN1KMOP3"),
    "in1kmop4": ("mo_nas.mo_nas", "IN1KMOP4"),
    "in1kmop5": ("mo_nas.mo_nas", "IN1KMOP5"),
    "in1kmop6": ("mo_nas.mo_nas", "IN1KMOP6"),
    "in1kmop7": ("mo_nas.mo_nas", "IN1KMOP7"),
    "in1kmop8": ("mo_nas.mo_nas", "IN1KMOP8"),
    "in1kmop9": ("mo_nas.mo_nas", "IN1KMOP9"),
    "mo_hopper_v2": ("morl.morl_problem", "MOHopperV2"),
    "mo_swimmer_v2": ("morl.morl_problem", "MOSwimmerV2"),
}


def get_problem(env_name, *args, **kwargs):
    name = env_name.lower()
    if name not in _PROBLEMS:
        raise ValueError("Unsupported paper task: {}".format(env_name))
    module, class_name = _PROBLEMS[name]
    problem_class = getattr(
        import_module("off_moo_bench.problem." + module), class_name
    )
    return problem_class(*args, **kwargs)
