from helpers2.vector import Packet, Vector, rand_point, origin_vec, com_type
from typing import Callable, List, Any, Union
import random
import multiprocessing as mp
import time
import logging

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)


class Particle(mp.Process):
    def __init__(
        self,
        init_position: Vector,
        init_velocity: Vector,
        cost: Callable[[Vector], float],
        acc_cog: float,
        acc_soc: float,
        id: Union[int, str],
        input_queue: mp.Queue,
        neighbours: List[mp.Queue],
        coordinator: bool,
        result: Any,
        steps: int,
        is_swarm_termination: bool,
        goal_dist: float,
        inertia: Callable[[], float],
    ) -> None:
        """
        Patricle constructor. In addition to the passed fields, it sets personal and global best values of the particle and store each as dict[str,Union[float,Vector]].
        Search field indicates the termiantion condition for the PSO, initially set to True, setting it False will terminate the algorithm

        :param init_position: initial position of the particle
        :param init_velocity: initial velocity of the particle
        :param cost: function of type Vector -> float which the swarm wants to minimize
        :param acc_cog: cognitive acceleration factor for PSO
        :param acc_soc: social acceleration factor for PSO
        :param id: identifier for the particle
        :param input_queue: process safe implementation of Queue (mp.Queue can be used or a class with same API should be used)
        :param neighbours: list of input queues of the neighbours of the particle
        :param coordinator: can be used to add custom logic for specific particles in the future, currently useless
        :param result: Manager().list() obeject that exists purely for debugging/plotting purposes
        :param steps: max number of iterations for PSO. if steps <= 0, repeat indefinetly. it is recommended to set steps to positive value
        :param goal_dist: additional parameter to specify a termination. algorithm terminates if the score is less than goal_dist. if goal_dist is set poorly and steps <= 0 algorithm might not terminates.
        :param is_swarm_termination: boolean to specify type of termination. if True, it is enough for only one particle to reach the desired area/point.
        :param inertia: callable that returns a float

        :returns: None
        """
        super(Particle, self).__init__()
        self.position = init_position
        self.velocity = init_velocity
        self.cost = cost
        self.acc_cog = acc_cog
        self.acc_soc = acc_soc
        score = cost(init_position)
        self.personal_best = {"Position": init_position, "Score": score}
        self.global_best = {"Position": init_position, "Score": score}
        self.id = id
        self.search = True
        self.goal_dist = goal_dist
        self.input_queue = input_queue
        self.neighbours = neighbours
        self.coordinator = coordinator
        self.steps = steps
        self.result = result
        self.is_swarm_termination = is_swarm_termination
        self.inertia = inertia

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.__str__()

    def transfer_packet(self, type: com_type = com_type.BROADCAST) -> None:
        """
        Method that transfers the packet to neighbours

        :param type: defines the type of communication

        :returns: None
        """
        match type:
            case com_type.BROADCAST:
                packet = Packet(
                    to_=-1,
                    from_=self.id,
                    poses=self.global_best,
                    stop=(not self.search) and self.is_swarm_termination,
                )
                for i in self.neighbours:
                    i.put(packet)
            case _:
                raise NotImplementedError

    def update_best(self) -> None:
        """
        Method that is used to update pbest and gbest values

        :returns: None
        """
        sp = self.cost(self.position)
        if sp < self.personal_best["Score"]:
            self.personal_best["Position"] = self.position
            self.personal_best["Score"] = sp
            self.result.append((self.id, "pbest", self.personal_best))
            if sp < self.global_best["Score"]:
                self.global_best["Position"] = self.position
                self.global_best["Score"] = sp
                self.result.append((self.id, "gbest", self.global_best))
                if sp < self.goal_dist:
                    self.search = False

    def update_pos(self, t: float) -> None:
        """
        Updates the position of the particle based on the time delta since last update, last position and velocity

        :param t: time delta since last update

        :returns:
        """
        self.position = self.position + self.velocity.scale(t)
        self.result.append((self.id, "pos", self.position))

    def update_pso(self) -> None:
        """
        PSO algorithm to update the velocity of the particle

        :returns: None
        """
        rand_1 = random.uniform(0, 1)
        rand_2 = random.uniform(0, 1)
        self.velocity = (
            self.velocity.scale(self.inertia())
            + (self.personal_best["Position"] - self.position).scale(
                self.acc_cog * rand_1
            )
            + (self.global_best["Position"] - self.position).scale(
                self.acc_soc * rand_2
            )
        )
        self.result.append((self.id, "velocity", self.velocity))

    def read_input_q(self) -> None:
        """
        TODO: explore get_nowait() method
        Reads the input queue to potentially update the global best

        :returns: None
        """
        best_in = {"Score": float("inf"), "Position": None}
        while not self.input_queue.empty():
            try:
                input = self.input_queue.get(timeout=1)
            except Exception as e:
                print(e)
                break
            if input.stop:
                self.search = False
            if input.poses["Score"] < best_in["Score"]:
                best_in = input.poses
        if best_in["Position"] is not None:
            if best_in["Score"] < self.global_best["Score"]:
                self.global_best = best_in
                self.result.append((self.id, "gbest", self.global_best))

    def run(self) -> None:
        """
        Loop that the process continuesly runs until a termiantion condition is met

        :returns: None
        """
        logging.info(f"Particle {self.id} is running")
        s = time.time()
        for _ in range(self.steps):
            if not self.search:
                logging.info(f"Early termination: Particle {self.id}")
                break
            self.transfer_packet()
            self.update_pos(time.time() - s)
            s = time.time()
            self.update_best()
            self.read_input_q()
            self.update_pso()
            self.update_pos(time.time() - s)
            s = time.time()
            self.update_best()
        logging.info(f"Particle {self.id} is terminating")


def pso_fun(x: Vector) -> float:
    return sum(map(lambda z: abs(z), x.p))


def inertia_fun() -> float:
    return random.uniform(0, 1) / 2 + 0.5

