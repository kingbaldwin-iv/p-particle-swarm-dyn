import json
from typing import Union, List
import numpy as np
from enum import Enum


class Vector:
    def __init__(self, p: list[float]) -> None:
        """
        :param p: coordiantes of the vector

        :returns: None
        """
        self.p = p
        self.d = len(p)

    def scale(self, s: float) -> "Vector":
        """
        :param s: scaling factor

        :returns: vector scaled by s
        """
        return Vector(p=[x * s for x in self.p])

    def distance(self, other: "Vector") -> float:
        """
        :param other: other vector to compare the distances

        :returns: euclidian distance between other and self
        """
        return sum([(i - v) ** 2 for i, v in zip(self.p, other.p)])

    def __add__(self, other: "Vector") -> "Vector":
        """
        :param other: other vector to add

        :returns: vector addition of self and other
        """
        return Vector(p=[i + v for i, v in zip(self.p, other.p)])

    def __sub__(self, other: "Vector") -> "Vector":
        """
        :param other: other vector to sub

        :returns: vector subtraction of other from self
        """
        return Vector(p=[i - v for i, v in zip(self.p, other.p)])

    def __str__(self) -> str:
        """
        :returns: str representation of vector
        """
        return json.dumps({"Dimension": self.d, "Vector": self.p}, indent=2)

    def __repr__(self) -> str:
        """
        :returns: appropriate str representation
        """
        return self.__str__()

    def __eq__(self, other):
        """
        :param other: other vector to compare with

        :retruns: True if euclidian distance between self and other is 0 else False
        """
        return all([i == v for i, v in zip(self.p, other.p)])


class Packet:
    def __init__(
        self,
        to_: Union[int, str],
        from_: Union[int, str],
        poses: dict[str, Union[Vector, dict[str, Union[Vector, float]]]],
        stop: bool,
    ) -> None:
        """
        TODO: to and from fields are not really used at this point
        :param to: the reciever of the packet (-1 implies broadcast)
        :param from: the sender of the packet
        :param poses: data field of the packet, depending on the kind of the oacket it can take various forms
        :param stop: paramtere to signal early termiantion to other particles

        :returns: None
        """
        self.to_ = to_
        self.from_ = from_
        self.poses = poses
        self.stop = stop


def random_unit_vector(d: int) -> List[float]:
    """
    :param d: dimension

    :returns: random unit vector of dimesnion d
    """
    vec = np.random.randn(d)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return random_unit_vector(d)
    return (vec / norm).tolist()


def rand_point(dist: float, dim: int = 2, base: Vector = Vector(p=[0, 0])) -> Vector:
    """
    Generates a random dim dimensional point which has euclidian distance dist to the point base

    :param dist: distance from base
    :param dim: dimension of the points
    :param base: point of origin

    :returns: random point on the hypersphere definied by middle point base, radius dist
    """
    if base.d != dim:
        raise ValueError
    v = Vector(p=random_unit_vector(dim)).scale(dist)
    return v + base


def origin_vec(d: int) -> Vector:
    """
    Generates the origin point for given dimension

    :param d: dimension

    :returns: origin point for the dimension d as instance of Vector
    """
    return Vector(p=[0 for _ in range(d)])


class com_type(Enum):
    BROADCAST = 1
    ROUTE = 2
