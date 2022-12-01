#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from simularium_models_util import ReaddyUtil


def run_one_timestep_readdy(mt_simulation):
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


def monomer_state_to_str(monomers):
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
    
    
def check_readdy_state(simulation, expected_monomers):
    test_monomers = ReaddyUtil.get_current_monomers(simulation.simulation.current_topologies)
    # raise Exception(monomer_state_to_str(test_monomers))
    assert_monomers_equal(test_monomers, expected_monomers, test_position=False)


def assert_monomers_equal(test_monomers, expected_monomers, test_position=False):
    """
    Assert two topologies (in monomer form) are equivalent
    """
    # check topology has the correct type_name
    # and contains the correct particle_ids (in any order, starting at any index)
    test_top_id = list(test_monomers["topologies"].keys())[0]
    exp_top_id = list(expected_monomers["topologies"].keys())[0]
    assert len(test_monomers["topologies"][test_top_id]["type_name"]) == len(
        expected_monomers["topologies"][exp_top_id]["type_name"]
    )
    test_particle_ids = test_monomers["topologies"][test_top_id]["particle_ids"]
    # pytest caches something that causes particle IDs to not always start at 0
    min_id = np.amin(np.array(test_particle_ids))
    assert len(test_particle_ids) == len(
        expected_monomers["topologies"][exp_top_id]["particle_ids"]
    )
    for particle_id in test_particle_ids:
        assert particle_id - min_id in expected_monomers["topologies"][exp_top_id]["particle_ids"], f"top particle IDs = {test_particle_ids}"
    for particle_id in expected_monomers["topologies"][exp_top_id]["particle_ids"]:
        assert particle_id + min_id in test_particle_ids
    # check the particle types, positions (optionally), and neighbors
    for particle_id in test_monomers["particles"]:
        test_particle = test_monomers["particles"][particle_id]
        exp_particle = expected_monomers["particles"][particle_id - min_id]
        assert test_particle["type_name"] == exp_particle["type_name"]
        neighbor_ids1 = test_particle["neighbor_ids"].copy()
        neighbor_ids1.sort()
        neighbor_ids2 = exp_particle["neighbor_ids"].copy()
        neighbor_ids2.sort()
        neighbor_ids2 = [nid + min_id for nid in neighbor_ids2]
        assert neighbor_ids1 == neighbor_ids2, f"Neighbors don't match for particle ID {particle_id - min_id}"
        if test_position:
            np.testing.assert_almost_equal(
                test_particle["position"], exp_particle["position"], decimal=2
            )


def assert_fibers_equal(topology_fibers1, topology_fibers2, test_position=False):
    """
    Assert two topologies (in fiber form) are equivalent
    """
    # check topology has the correct type_name
    # and contains the correct points (in order)
    assert len(topology_fibers1) == len(topology_fibers2)
    for f in range(len(topology_fibers1)):
        assert topology_fibers1[f].type_name == topology_fibers2[f].type_name
        assert len(topology_fibers1[f].points) == len(topology_fibers2[f].points)
        for p in range(len(topology_fibers1[f].points)):
            np.testing.assert_allclose(
                topology_fibers1[f].points[p], topology_fibers2[f].points[p]
            )
