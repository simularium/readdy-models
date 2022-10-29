#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import math

from simularium_models_util.microtubules import (
    MicrotubulesSimulation
)
from simularium_models_util import ReaddyUtil

from simularium_models_util.tests.conftest import assert_monomers_equal

default_parameters = {
    "name": "",
    "total_steps": 10,
    "time_step": 0.1,
    "box_size": [300.0, 300.0, 300.0],  # nm
    "temperature_C": 37.0,  # from Pollard experiments
    "viscosity": 8.1,  # cP, viscosity in cytoplasm
    "force_constant": 75.0,
    "grow_reaction_distance": 1.0,  # nm
    "attach_reaction_distance": 1.0,  # nm
    "n_cpu": 4,
    "tubulin_radius": 2.,
    "protofilament_growth_GTP_rate": 0.,  # 1/ns
    "protofilament_growth_GDP_rate": 0.,  # 1/ns
    "protofilament_shrink_GTP_rate": 0.,  # 1/ns
    "protofilament_shrink_GDP_rate": 0.,  # 1/ns
    "ring_attach_GTP_rate": 0.,  # 1/ns
    "ring_attach_GDP_rate": 0.,  # 1/ns
    "ring_detach_GTP_rate": 0.,  # 1/ns
    "ring_detach_GDP_rate": 0.,  # 1/ns
    "hydrolyze_rate": 0.,  # 1/ns
    "verbose": True,
}


def tubulin_is_bent(ring_ix, filament_ix, ring_connections):
    """
    Is the tubulin in bent state?
    Bent means it is missing at least one ring edge
    """
    return (
        False in ring_connections[ring_ix + 1, filament_ix:filament_ix + 2]
    )


def tubulin_has_sites(ring_ix, filament_ix, ring_connections):
    """
    Does the tubulin need to have sites?
    Based on whether it and its neighbors toward + and - ends
    are fully crosslinked to their lateral neighbors
    """
    return (
        False in ring_connections[ring_ix:ring_ix + 3, filament_ix:filament_ix + 2]
    )


def get_edge(pid1, pid2):
    """
    Create a tuple of particle ids for the edge
    
    ReaDDy seems to sometimes not add edges if the first int 
    in the tuple is greater than the second. Why???
    """
    # print(f"Added edge: ({pid1}, {pid2})")
    return (pid1, pid2)
    if pid1 < pid2:
        return (pid1, pid2)
    else:
        return (pid2, pid1)


def add_init_state(readdy_simulation, ring_connections, topology_type):
    """
    Create patch of tubulins with given types and connections
    """
    particle_dims = ring_connections.shape 
    n_rings = particle_dims[0] - 2  # since there's two less rings than filament edges
    n_filaments = particle_dims[1] - 1  # since there's one less filament than ring edges
    n_tubulin = n_rings * n_filaments
    types = []
    site_types = []
    positions = []
    site_positions = []
    edges = []
    d_ring = np.array([4., 0., 0.])
    d_filament = np.array([0., 0., 4.])
    out_string = "\n"
    for filament_ix in range(n_filaments):
        for ring_ix in range(n_rings):
            # each tubulin
            tub_type = "A" if ring_ix % 2 == 0 else "B"
            bent_type = "bent_" if tubulin_is_bent(ring_ix, filament_ix, ring_connections) else ""
            number1 = ring_ix % 3 + 1
            number2 = (filament_ix + math.floor(ring_ix / 3)) % 3 + 1
            has_sites = tubulin_has_sites(ring_ix, filament_ix, ring_connections)
            out_string += f"tub [{ring_ix}, {filament_ix}] : {tub_type}, {number1}_{number2}, sites = {has_sites}\n"
            types.append(f"tubulin{tub_type}#GTP_{bent_type}{number1}_{number2}")
            positions.append(ring_ix * d_ring + filament_ix * d_filament)
            tubulin_id = filament_ix * n_rings + ring_ix
            # bond to minus end tubulin
            if ring_ix > 0:
                minus_id = filament_ix * n_rings + ring_ix - 1
                edges.append(get_edge(tubulin_id, minus_id))
            # bond to plus end tubulin
            if ring_ix < n_rings - 1:
                plus_id = filament_ix * n_rings + ring_ix + 1
                edges.append(get_edge(tubulin_id, plus_id))
            # bond to prev ring tubulin
            if filament_ix > 0 and ring_connections[ring_ix + 1, filament_ix]:
                prev_id = (filament_ix - 1) * n_rings + ring_ix
                edges.append(get_edge(tubulin_id, prev_id))
            # bond to next ring tubulin
            if filament_ix < n_filaments - 1 and ring_connections[ring_ix + 1, filament_ix + 1]:
                next_id = (filament_ix + 1) * n_rings + ring_ix
                edges.append(get_edge(tubulin_id, next_id))
            # sites
            if has_sites:
                tublin_position = positions[len(positions) - 1]
                site_out_id = n_tubulin + len(site_types)
                for site_ix in range(5):
                    site_id = site_out_id + site_ix
                    edges.append(get_edge(site_id, tubulin_id))  # edge to tubulin
                    if site_ix == 0:
                        d_site = np.array([0., 2., 0.])
                        site_types.append("site#out")
                    else:
                        d_site = 0.5 * (d_ring if site_ix > 2 else d_filament)
                        d_site *= -1 if site_ix % 2 == 1 else 1
                        site_types.append(f"site#{site_ix}")
                        edges.append(get_edge(site_id, site_out_id))  # edge to site#out
                        # if site_ix > 2:  # diagonal edges between sites
                        #     edges.append((site_id, site_id - (site_ix - 2))) 
                        #     edges.append((site_id, site_id - (site_ix - 1)))
                    site_positions.append(tublin_position + d_site)
                # # site bonds to minus end tubulin sites
                # if ring_ix > 0 and tubulin_has_sites(ring_ix - 1, filament_ix, ring_connections):
                #     minus_site_out_id = site_out_id - 5
                #     edges.append((site_out_id + 1, minus_site_out_id + 1))  # sites 1
                #     edges.append((site_out_id + 2, minus_site_out_id + 2))  # sites 2
                #     edges.append((site_out_id + 3, minus_site_out_id + 4))  # sites 3 and 4
                # # site bonds to plus end tubulin sites
                # if ring_ix < n_rings - 1 and tubulin_has_sites(ring_ix + 1, filament_ix, ring_connections):
                #     plus_site_out_id = site_out_id + 5
                #     edges.append((site_out_id + 1, plus_site_out_id + 1))  # sites 1
                #     edges.append((site_out_id + 2, plus_site_out_id + 2))  # sites 2
                #     edges.append((site_out_id + 4, plus_site_out_id + 3))  # sites 3 and 4
    microtubule = readdy_simulation.add_topology(
        topology_type, types + site_types, np.array(positions + site_positions)
    )
    for edge in edges:
        microtubule.get_graph().add_edge(edge[0], edge[1])
        
        
def create_simulation(ring_connections, topology_type):
    """
    create simulation and add initial particles
    """
    mt_simulation = MicrotubulesSimulation(default_parameters, just_bonds=True)
    add_init_state(mt_simulation.simulation, ring_connections, topology_type)
    return mt_simulation


def run_readdy(mt_simulation):
    # setup readdy functions
    timestep = 0.1
    readdy_actions = mt_simulation.simulation._actions
    init = readdy_actions.initialize_kernel()
    # diffuse = readdy_actions.integrator_euler_brownian_dynamics(timestep)
    calculate_forces = readdy_actions.calculate_forces()
    create_nl = readdy_actions.create_neighbor_list(mt_simulation.system.calculate_max_cutoff().magnitude)
    update_nl = readdy_actions.update_neighbor_list()
    react = readdy_actions.reaction_handler_uncontrolled_approximation(timestep)
    observe = readdy_actions.evaluate_observables()
    # run simulation
    init()
    create_nl()
    calculate_forces()
    update_nl()
    observe(0)
    update_nl()
    react()        
    update_nl()
    calculate_forces()
    observe(1)
    
    
def state_to_str(monomers):
    result = "topologies:\n"
    for top_id in monomers["topologies"]:
        top = monomers["topologies"][top_id]
        result += f"  {top_id} : {top}\n"
    result += "particles:\n"
    for particle_id in monomers["particles"]:
        particle = monomers["particles"][particle_id]
        type_name = particle["type_name"]
        neighbor_ids = particle["neighbor_ids"]
        result += f"  {particle_id} : {type_name}, {neighbor_ids}\n"
    return result
    
    
def check_readdy_state(mt_simulation, expected_monomers):
    test_monomers = ReaddyUtil.get_current_monomers(mt_simulation.simulation.current_topologies)
    # raise Exception(state_to_str(test_monomers))
    assert_monomers_equal(test_monomers, expected_monomers, test_position=False)


@pytest.mark.parametrize(
    "ring_connections, topology_type, expected_monomers",
    [
        # 5 rings x 4 filaments, fully crosslinked
        # (
        #     # ring connections: + 2 rings, + 1 filament edges
        #     np.array([
        #     # f       0     1     2     3           r
        #         [True, True, True, True, True], 
        #         [True, True, True, True, True],   # 0
        #         [True, True, True, True, True],   # 1
        #         [True, True, True, True, True],   # 2
        #         [True, True, True, True, True],   # 3
        #         [True, True, True, True, True],   # 4
        #         [True, True, True, True, True], 
        #     ]),
        #     "Microtubule",
        #     {
        #         "topologies": {
        #             0: {
        #                 "type_name": "Microtubule",
        #                 "particle_ids": range(20),
        #             }
        #         },
        #         "particles": {
        #             0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
        #             1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
        #             2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7]},
        #             3: {"unique_id": 3, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 8]},
        #             4: {"unique_id": 4, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [3, 9]},
        #             5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
        #             6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11]},
        #             7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [6, 8, 2, 12]},
        #             8: {"unique_id": 8, "type_name": "tubulinB#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 3, 13]},
        #             9: {"unique_id": 9, "type_name": "tubulinA#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [8, 4, 14]},
        #             10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
        #             11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16]},
        #             12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 7, 17]},
        #             13: {"unique_id": 13, "type_name": "tubulinB#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 8, 18]},
        #             14: {"unique_id": 14, "type_name": "tubulinA#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [13, 9, 19]},
        #             15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
        #             16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11]},
        #             17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 12]},
        #             18: {"unique_id": 18, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 13]},
        #             19: {"unique_id": 19, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [18, 14]},
        #         },
        #     }
        # ),
        # 5 rings x 4 filaments, missing one ring edge
        (
            # ring connections: + 2 rings, + 1 filament edges
            np.array([
            # f       0     1     2     3           r
                [True, True, True, True, True], 
                [True, True, True, True, True],   # 0
                [True, True, True, True, True],   # 1
                [True, True, False, True, True],  # 2
                [True, True, True, True, True],   # 3
                [True, True, True, True, True],   # 4
                [True, True, True, True, True], 
            ]), # 5 rings, only ring bonds shown, filament bonds always true
            "Microtubule",
            {
                "topologies": {
                    0: {
                        "type_name": "Microtubule",
                        "particle_ids": range(50),
                    }
                },
                # type_name format: particle_type_{filament}_{ring}
                "particles": {
                    0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
                    1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
                    2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7]},
                    3: {"unique_id": 3, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 8]},
                    4: {"unique_id": 4, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [3, 9]},
                    5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
                    6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11]},
                    7: {"unique_id": 7, "type_name": "tubulinA#GTP_bent_3_2", "position": np.zeros(3), "neighbor_ids": [6, 8, 2]},
                    8: {"unique_id": 8, "type_name": "tubulinB#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 3, 13]},
                    9: {"unique_id": 9, "type_name": "tubulinA#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [8, 4, 14]},
                    10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
                    11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16]},
                    12: {"unique_id": 12, "type_name": "tubulinA#GTP_bent_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 17]},
                    13: {"unique_id": 13, "type_name": "tubulinB#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 8, 18]},
                    14: {"unique_id": 14, "type_name": "tubulinA#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [13, 9, 19]},
                    15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
                    16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11]},
                    17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 12]},
                    18: {"unique_id": 18, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 13]},
                    19: {"unique_id": 19, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    20: {"unique_id": 20, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    21: {"unique_id": 21, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    22: {"unique_id": 22, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    23: {"unique_id": 23, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    24: {"unique_id": 24, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    25: {"unique_id": 25, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    26: {"unique_id": 26, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    27: {"unique_id": 27, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    28: {"unique_id": 28, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    29: {"unique_id": 29, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    30: {"unique_id": 30, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    31: {"unique_id": 31, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    32: {"unique_id": 32, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    33: {"unique_id": 33, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    34: {"unique_id": 34, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    35: {"unique_id": 35, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    36: {"unique_id": 36, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    37: {"unique_id": 37, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    38: {"unique_id": 38, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    39: {"unique_id": 39, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    40: {"unique_id": 40, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    41: {"unique_id": 41, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    42: {"unique_id": 42, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    43: {"unique_id": 43, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    44: {"unique_id": 44, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    45: {"unique_id": 45, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    46: {"unique_id": 46, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    47: {"unique_id": 47, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    48: {"unique_id": 48, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    49: {"unique_id": 49, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    50: {"unique_id": 50, "type_name": "site#", "position": np.zeros(3), "neighbor_ids": [18, 14]}                    
                },
            }
        ),
    ],
)
def test_attach(ring_connections, topology_type, expected_monomers):
    mt_simulation = create_simulation(ring_connections, topology_type)
    run_readdy(mt_simulation)
    check_readdy_state(mt_simulation, expected_monomers)
    del mt_simulation
