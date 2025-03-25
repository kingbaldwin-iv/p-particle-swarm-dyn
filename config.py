from enum import Enum
import multiprocessing as mp
from typing import Any, Callable, List
from helpers2.vector import Vector, origin_vec, rand_point
import random
from functools import partial
import colorsys
import numpy as np


def generate_n_colors(N):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples


class problem_type(Enum):
    MEET_AT_POINT_X = 1


class topology_type(Enum):
    WHEEL = 1
    CYCLE = 2
    STAR = 3
    FULLY_CONNECTED = 4


def deafult_inertia() -> float:
    return random.uniform(0, 1) / 2 + 0.5
def deafult_inertia_high() -> float:
    return 0.9
def deafult_inertia_low() -> float:
    return 0.5

def diff_1norm(v1: Vector, v2: Vector) -> float:
    return sum(map(lambda i: i**2, (v1 - v2).p))


def problem_types_to_func(**kwargs: dict[str, Any]) -> Callable[[Vector], float]:
    match kwargs["type"]:
        case problem_type.MEET_AT_POINT_X:
            return partial(diff_1norm, kwargs["point"])
        case _:
            raise NotImplementedError


def randomly_generated_initial_positions(
    num: int = mp.cpu_count() - 1,
    dim: int = 2,
    max_dist: int = 100,
    min_dist: int = 1,
    groups: int = 1,
    base: Vector = origin_vec(2),
) -> List[List[Vector]]:
    """
    :param num: number of points to generate randomly
    :param dim: dimension of the points
    :param max_dist: radius of hypersphere in which points can be picked
    :param min_dist: radius of hypersphere in which points can't be picked

    :returns: groups many lists, in total num many, dim dimensional random vectors
    """
    bases = [
        rand_point(dist=random.uniform(min_dist, max_dist), dim=2, base=base)
        for _ in range(groups)
    ]
    poses = np.array_split([0] * num, groups)
    poses = [
        list(
            map(
                lambda _: rand_point(
                    dist=random.uniform(min_dist, max_dist // 10), dim=2, base=b
                ),
                pl,
            )
        )
        for pl, b in zip(poses, bases)
    ]
    return poses


def randomly_generated_init_vol(
    ps: List[Vector] = randomly_generated_initial_positions()[0], groups: int = 1
) -> List[List[Vector]]:
    """
    TODO: Algorithm has no basis, it is just a pseudo random way to generate some vectors.
    :param ps: list of vectors

    :returns: list of initial velocities
    """
    s = Vector(p=[x / len(ps) for x in sum(ps, origin_vec(ps[0].d)).p])
    return list(np.array_split([s - i for i in ps], groups))


"""
:param initial_positions: if calling random generation function, make sure groups param is set to groups["group_count"]
:param initial_velocities: if calling random generation function, make sure groups param is set to groups["group_count"]
"""
params = {
    "problem_type": {"type": problem_type.MEET_AT_POINT_X, "point": rand_point(dist=random.uniform(0,8))},
    "topology": topology_type.CYCLE,
    "input_buffer_size": -1,
    "number_of_particles": mp.cpu_count() - 1,
    "group_count": 2,
    "initial_positions": randomly_generated_initial_positions(groups=2),
    "initial_velocities": randomly_generated_initial_positions(groups=2),
    "cognitive_acc": 2.0,
    "social_acc": 2.0,
    "inertia": deafult_inertia,
    "goal_dist": 0.0,
    "is_swarm_termination": False,
    "steps": 1000,
    "plot": True,
    "leader_topology": topology_type.CYCLE
}

