#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np


from simularium_models_util.tests.conftest import run_readdy, check_readdy_state
from simularium_models_util.tests.microtubules.microtubules_conftest import create_microtubules_simulation


@pytest.mark.parametrize(
    "ring_connections, topology_type, new_edge_tub_ix, expected_monomers",
    [
        # 5 rings x 4 filaments, fully crosslinked, no reaction
        (
            # ring connections: + 2 rings, + 1 filament edges
            # (5 rings, only ring bonds shown, filament bonds always true)
            np.array([
            # f       0     1     2     3           r
                [True, True, True, True, True], 
                [True, True, True, True, True],   # 0
                [True, True, True, True, True],   # 1
                [True, True, True, True, True],   # 2
                [True, True, True, True, True],   # 3
                [True, True, True, True, True],   # 4
                [True, True, True, True, True], 
            ]),
            "Microtubule",
            None,  # no reaction to form edges
            {
                "topologies": {
                    0: {
                        "type_name": "Microtubule",
                        "particle_ids": range(20),
                    }
                },
                "particles": {
                    0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
                    1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
                    2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7]},
                    3: {"unique_id": 3, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 8]},
                    4: {"unique_id": 4, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [3, 9]},
                    5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
                    6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11]},
                    7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [6, 8, 2, 12]},
                    8: {"unique_id": 8, "type_name": "tubulinB#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 3, 13]},
                    9: {"unique_id": 9, "type_name": "tubulinA#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [8, 4, 14]},
                    10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
                    11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16]},
                    12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 7, 17]},
                    13: {"unique_id": 13, "type_name": "tubulinB#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 8, 18]},
                    14: {"unique_id": 14, "type_name": "tubulinA#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [13, 9, 19]},
                    15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
                    16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11]},
                    17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 12]},
                    18: {"unique_id": 18, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 13]},
                    19: {"unique_id": 19, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                },
            }
        ),
        # 5 rings x 4 filaments, missing one ring edge, no reaction
        (
            np.array([
            # f       0     1     2     3           r
                [True, True, True, True, True], 
                [True, True, True, True, True],   # 0
                [True, True, True, True, True],   # 1
                [True, True, False, True, True],  # 2
                [True, True, True, True, True],   # 3
                [True, True, True, True, True],   # 4
                [True, True, True, True, True], 
            ]), 
            "Microtubule",
            None,  # no reaction to form edges
            {
                "topologies": {
                    0: {
                        "type_name": "Microtubule",
                        "particle_ids": range(50),
                    }
                },
                # type_name format: particle_type_{filament}_{ring}
                "particles": {
                    # particle ids 6,7,8, 11,12,13 will have sites
                    # available site types are: 
                    #   out = normal to MT surface
                    #   1 = towards the previous protofilament (same ring)
                    #   2 = towards the next protofilament (same ring)
                    #   3 = towards the minus end (same protofilament)
                    #   4 = towards the plus end (same protofilament)
                    0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
                    1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
                    2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7]},
                    3: {"unique_id": 3, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 8]},
                    4: {"unique_id": 4, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [3, 9]},
                    5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
                    6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11, 20, 21, 22, 23, 24]},
                    7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [6, 8, 2, 25, 26, 27, 28, 29]},
                    8: {"unique_id": 8, "type_name": "tubulinB#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 3, 13, 30, 31, 32, 33, 34]},
                    9: {"unique_id": 9, "type_name": "tubulinA#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [8, 4, 14]},
                    10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
                    11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16, 35, 36, 37, 38, 39]},
                    12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 17, 40, 41, 42, 43, 44]},
                    13: {"unique_id": 13, "type_name": "tubulinB#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 8, 18, 45, 46, 47, 48, 49]},
                    14: {"unique_id": 14, "type_name": "tubulinA#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [13, 9, 19]},
                    15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
                    16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11]},
                    17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 12]},
                    18: {"unique_id": 18, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 13]},
                    19: {"unique_id": 19, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                    20: {"unique_id": 20, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [6, 21, 22, 23, 24, 25]},
                    21: {"unique_id": 21, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [6, 20, 23, 24, 26]},
                    22: {"unique_id": 22, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [6, 20, 23, 24, 27]},
                    23: {"unique_id": 23, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [6, 20, 21, 22]},
                    24: {"unique_id": 24, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [6, 20, 21, 22, 28]},
                    25: {"unique_id": 25, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [7, 20, 26, 27, 28, 29, 30]},
                    26: {"unique_id": 26, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [7, 21, 25, 28, 29, 31]},
                    27: {"unique_id": 27, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [7, 22, 25, 28, 29, 32]},
                    28: {"unique_id": 28, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [7, 25, 26, 27, 24]},
                    29: {"unique_id": 29, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [7, 25, 26, 27, 33]},
                    30: {"unique_id": 30, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [8, 25, 31, 32, 33, 34]},
                    31: {"unique_id": 31, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [8, 26, 30, 33, 34]},
                    32: {"unique_id": 32, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [8, 27, 30, 33, 34]},
                    33: {"unique_id": 33, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [8, 30, 31, 32, 29]},
                    34: {"unique_id": 34, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [8, 30, 31, 32]},
                    35: {"unique_id": 35, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [11, 36, 37, 38, 39, 40]},
                    36: {"unique_id": 36, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [11, 35, 38, 39, 41]},
                    37: {"unique_id": 37, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [11, 35, 38, 39, 42]},
                    38: {"unique_id": 38, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [11, 35, 36, 37]},
                    39: {"unique_id": 39, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [11, 35, 36, 37, 43]},
                    40: {"unique_id": 40, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [12, 35, 41, 42, 43, 44, 45]},
                    41: {"unique_id": 41, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [12, 36, 40, 43, 44, 46]},
                    42: {"unique_id": 42, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [12, 37, 40, 43, 44, 47]},
                    43: {"unique_id": 43, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [12, 40, 41, 42, 39]},
                    44: {"unique_id": 44, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [12, 40, 41, 42, 48]},
                    45: {"unique_id": 45, "type_name": "site#out", "position": np.zeros(3), "neighbor_ids": [13, 40, 46, 47, 48, 49]},
                    46: {"unique_id": 46, "type_name": "site#1", "position": np.zeros(3), "neighbor_ids": [13, 41, 45, 48, 49]},
                    47: {"unique_id": 47, "type_name": "site#2", "position": np.zeros(3), "neighbor_ids": [13, 42, 45, 48, 49]},
                    48: {"unique_id": 48, "type_name": "site#3", "position": np.zeros(3), "neighbor_ids": [13, 45, 46, 47, 44]},
                    49: {"unique_id": 49, "type_name": "site#4", "position": np.zeros(3), "neighbor_ids": [13, 45, 46, 47]}               
                },
            }
        ),
        # 5 rings x 4 filaments, missing one ring edge, react to attach missing edge
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
            ]), 
            "Microtubule#Attaching",
            [7, 12],  # add the one missing edge
            {
                "topologies": {
                    0: {
                        "type_name": "Microtubule",
                        "particle_ids": range(20),
                    }
                },
                "particles": {
                    0: {"unique_id": 0, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [1, 5]},
                    1: {"unique_id": 1, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [0, 2, 6]},
                    2: {"unique_id": 2, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [1, 3, 7]},
                    3: {"unique_id": 3, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [2, 4, 8]},
                    4: {"unique_id": 4, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [3, 9]},
                    5: {"unique_id": 5, "type_name": "tubulinA#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [6, 0, 10]},
                    6: {"unique_id": 6, "type_name": "tubulinB#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [5, 7, 1, 11]},
                    7: {"unique_id": 7, "type_name": "tubulinA#GTP_3_2", "position": np.zeros(3), "neighbor_ids": [6, 8, 2, 12]},
                    8: {"unique_id": 8, "type_name": "tubulinB#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [7, 9, 3, 13]},
                    9: {"unique_id": 9, "type_name": "tubulinA#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [8, 4, 14]},
                    10: {"unique_id": 10, "type_name": "tubulinA#GTP_1_3", "position": np.zeros(3), "neighbor_ids": [11, 5, 15]},
                    11: {"unique_id": 11, "type_name": "tubulinB#GTP_2_3", "position": np.zeros(3), "neighbor_ids": [10, 12, 6, 16]},
                    12: {"unique_id": 12, "type_name": "tubulinA#GTP_3_3", "position": np.zeros(3), "neighbor_ids": [11, 13, 7, 17]},
                    13: {"unique_id": 13, "type_name": "tubulinB#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [12, 14, 8, 18]},
                    14: {"unique_id": 14, "type_name": "tubulinA#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [13, 9, 19]},
                    15: {"unique_id": 15, "type_name": "tubulinA#GTP_1_1", "position": np.zeros(3), "neighbor_ids": [16, 10]},
                    16: {"unique_id": 16, "type_name": "tubulinB#GTP_2_1", "position": np.zeros(3), "neighbor_ids": [15, 17, 11]},
                    17: {"unique_id": 17, "type_name": "tubulinA#GTP_3_1", "position": np.zeros(3), "neighbor_ids": [16, 18, 12]},
                    18: {"unique_id": 18, "type_name": "tubulinB#GTP_1_2", "position": np.zeros(3), "neighbor_ids": [17, 19, 13]},
                    19: {"unique_id": 19, "type_name": "tubulinA#GTP_2_2", "position": np.zeros(3), "neighbor_ids": [18, 14]},
                },
            }
        ),
    ],
)
def test_attach(ring_connections, topology_type, new_edge_tub_ix, expected_monomers):
    total_steps = 1
    mt_simulation = create_microtubules_simulation(ring_connections, topology_type, new_edge_tub_ix)
    run_readdy(total_steps, mt_simulation.system, mt_simulation.simulation)
    check_readdy_state(mt_simulation, expected_monomers)
    del mt_simulation
