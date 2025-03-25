import config
import multiprocessing as mp
from particle import Particle, Vector
from typing import List, Callable, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class swarm_manager:
    def __init__(
        self,
        problem_type_dict: Dict[str, Union[config.problem_type, Any]],
        topology: config.topology_type,
        buffer_size: int,
        leader_topology: config.topology_type,
        particle_count: int,
        init_positions: List[List[Vector]],
        init_velocities: List[List[Vector]],
        acc_cog: float,
        acc_soc: float,
        group_count: int,
        steps: int,
        is_swarm_termination: bool,
        goal_dist: float,
        inertia: Callable[[], float],
        plot: bool,
    ) -> None:
        self.res_manager = mp.Manager()
        self.res_list = self.res_manager.list()
        self.problem_type_dict = problem_type_dict
        self.particle_count = particle_count
        self.cost = config.problem_types_to_func(**problem_type_dict)
        self.particle_list_list = list()
        for pv, idf in zip(
            zip(init_positions, init_velocities), range(1, group_count + 1)
        ):
            ids = list(
                map(
                    lambda x: str(x[0]) + "." + str(x[1]),
                    zip([idf] * len(pv[0]), range(len(pv[0]))),
                )
            )
            subl = list()
            for co in range(len(pv[0])):
                subl.append(
                    Particle(
                        init_position=pv[0][co],
                        init_velocity=pv[1][co],
                        cost=self.cost,
                        acc_cog=acc_cog,
                        acc_soc=acc_soc,
                        id=ids[co],
                        input_queue=mp.Queue(buffer_size),
                        neighbours=[],
                        coordinator=False,
                        result=self.res_list,
                        steps=steps,
                        is_swarm_termination=is_swarm_termination,
                        goal_dist=goal_dist,
                        inertia=inertia,
                    )
                )
            self.particle_list_list.append(subl)
        self.edges = list()
        self.coloring = dict(
            sum(
                [
                    list(map(lambda u: (u.id, v), k))
                    for k, v in zip(
                        self.particle_list_list, config.generate_n_colors(group_count)
                    )
                ],
                [],
            )
        )
        self.leaders = [i[0] for i in self.particle_list_list]
        self.leader_topology = leader_topology
        match leader_topology:
            case config.topology_type.FULLY_CONNECTED:
                for i in self.leaders:
                    for j in self.leaders:
                        if j.id != i.id:
                            i.neighbours.append(j.input_queue)
                            j.neighbours.append(i.input_queue)
                            self.edges.append((i.id,j.id))
                            self.edges.append((j.id,i.id))
            case config.topology_type.CYCLE:
                for i in range(group_count):
                    other_index = (i + 1) % group_count
                    self.leaders[i].neighbours.append(self.leaders[other_index].input_queue)
                    self.leaders[other_index].neighbours.append(self.leaders[i].input_queue)
                    self.edges.append((self.leaders[i].id,self.leaders[other_index].id))
                    self.edges.append((self.leaders[other_index].id,self.leaders[i].id))
            case _:
                raise NotImplementedError
        match topology:
            case config.topology_type.WHEEL:
                for i in self.particle_list_list:
                    for j in range(len(i)):
                        other_index = (j + 1) % len(i)
                        i[j].neighbours.append(i[other_index].input_queue)
                        i[other_index].neighbours.append(i[j].input_queue)
                        self.edges.append((i[j].id, i[other_index].id))
                        self.edges.append((i[other_index].id, i[j].id))
                        if j != 0:
                            i[j].neighbours.append(i[0].input_queue)
                            i[0].neighbours.append(i[j].input_queue)
                            self.edges.append((i[0].id, i[j].id))
                            self.edges.append((i[j].id, i[0].id))
            case config.topology_type.CYCLE:
                for i in self.particle_list_list:
                    for j in range(len(i)):
                        other_index = (j + 1) % len(i)
                        i[j].neighbours.append(i[other_index].input_queue)
                        i[other_index].neighbours.append(i[j].input_queue)
                        self.edges.append((i[j].id, i[other_index].id))
                        self.edges.append((i[other_index].id, i[j].id))
            case config.topology_type.STAR:
                for i in self.particle_list_list:
                    for j in range(1, len(i)):
                        i[j].neighbours.append(i[0].input_queue)
                        i[0].neighbours.append(i[j].input_queue)
                        self.edges.append((i[j].id, i[0].id))
                        self.edges.append((i[0].id, i[j].id))
            case config.topology_type.FULLY_CONNECTED:
                for i in self.particle_list_list:
                    for j in range(len(i)):
                        for k in range(len(i)):
                            if i[j].id != i[k].id:
                                i[j].neighbours.append(i[k].input_queue)
                                i[k].neighbours.append(i[j].input_queue)
                                self.edges.append((i[j].id, i[k].id))
                                self.edges.append((i[k].id, i[j].id))
            case _:
                raise NotImplementedError
        self.plot = plot
        if plot:
            import networkx as nx
            f,ax = plt.subplots(1,2)
            G = nx.DiGraph()
            G.add_edges_from(self.edges)
            pos = nx.spring_layout(G)
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="lightblue",
                edge_color="gray",
                node_size=100,
                font_size=8,
                ax=ax[1]
            )
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(u, v): f"{u}->{v}" for u, v in self.edges}, ax=ax[1]
            )
            self.fig = f
            self.ax = ax

    def release(self) -> None:
        for i in self.particle_list_list:
            for j in i:
                j.start()
        for i in self.particle_list_list:
            for j in i:
                j.join(timeout=5)
        for i in self.particle_list_list:
            for j in i:
                j.input_queue.close()
                j.input_queue.join_thread()

    def plot_and_finish(self) -> None:
        if not self.plot:
            f,ax = plt.subplots()
        else:
            f = self.fig
            ax = self.ax[0]
        it = set(map(lambda x: x.id, sum(self.particle_list_list, [])))
        for i in tqdm(it, desc="Plotting"):
            mmx = list(filter(lambda x: x[1] == "pos" and x[0] == i, self.res_list))
            mmx = [dw[2].p for dw in mmx]
            xx = [dw[0] for dw in mmx]
            yx = [dw[1] for dw in mmx]
            col = self.coloring[i]
            ax.plot(xx, yx, color=col, alpha=0.5)
            ax.scatter(xx[0], yx[0], color=col)
            ax.annotate(i,(xx[0],yx[0]))
        ax.scatter(
            self.problem_type_dict["point"].p[0],
            self.problem_type_dict["point"].p[1],
            s=100,
            marker="x",
        )
        plt.show()


def swarm_from_config(params: dict[str,Any] = config.params) -> swarm_manager:
    swarm = swarm_manager(
        problem_type_dict=params["problem_type"],
        leader_topology=params["leader_topology"],
        topology=params["topology"],
        buffer_size=params["input_buffer_size"],
        particle_count=params["number_of_particles"],
        group_count=params["group_count"],
        steps=params["steps"],
        is_swarm_termination=params["is_swarm_termination"],
        goal_dist=params["goal_dist"],
        inertia=params["inertia"],
        init_positions=params["initial_positions"],
        init_velocities=params["initial_velocities"],
        acc_cog=params["cognitive_acc"],
        acc_soc=params["social_acc"],
        plot=params["plot"],
    )
    return swarm


def run_swarm_from_config(param: dict[str,Any] = config.params) -> None:
    s = swarm_from_config(param)
    s.release()
    s.plot_and_finish()


if __name__ == "__main__":
    run_swarm_from_config()
