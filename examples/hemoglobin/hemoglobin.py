#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import math
import copy

import numpy as np
import readdy
from simulariumio import (
    TrajectoryConverter,
    MetaData,
    DisplayData,
    DISPLAY_TYPE,
    UnitData,
    DimensionData,
)
from simulariumio.readdy import (
    ReaddyConverter,
    ReaddyData,
)
from scipy.spatial.transform import Rotation

from simularium_readdy_models import ReaddyUtil


TIMESTEP = 0.1 # ns
BOX_SIZE = 20. # nm
N_HEMOGLOBIN_SIDE = 1 # will be cubed
N_OXYGEN = 10 # 118 total, most start bound to Hb
N_CO = 21 # serious poisoning
HEMOGLOBIN_RADIUS = 1.5 # nm
OXYGEN_RADIUS = 0.15 # nm
CO_RADIUS = 0.06 # nm
TEMPERATURE = 37. + 273. # C -> K
FORCE_CONSTANT = 100.
HEMOGLOBIN_ANGLE = np.deg2rad(120.)
O2_BINDING_RATE = 0.5  # ~81 events in 30ms
O2_UNBINDING_RATE = 0.0001  # ~81 events in 30ms
START_PERCENT_O2_BOUND = 98.
CO_BINDING_RATE = 0.001  # TODO ~18 events in 30ms
CO_UNBINDING_RATE = 0.001  # TODO 1 event in 30ms

n_binding_events = 0
n_unbinding_events = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy hemoglobin binding simulation"
    )
    parser.add_argument(
        "output_name", help="name of output files"
    )
    parser.add_argument(
        "total_steps", help="how many timesteps to calculate"
    )
    parser.add_argument('--save_pickle', action=argparse.BooleanOptionalAction)
    parser.set_defaults(save_pickle=False)
    return parser.parse_args()

def add_types_and_potentials(system):
    viscosity = 8.1  # cP
    margin = 1.
    box_potential_size = np.array(3 * [BOX_SIZE - 2 * margin])
    system.topologies.add_type("Hemoglobin")
    system.topologies.add_type("Oxygen")
    system.topologies.add_type("Associating")
    system.topologies.add_type("Dissociating1")
    system.topologies.add_type("Dissociating2")
    system.add_species("oxygen#remove", 0)
    particles = {
        "hemoglobin" : HEMOGLOBIN_RADIUS,
        "hemoglobin#1" : HEMOGLOBIN_RADIUS,
        "hemoglobin#2" : HEMOGLOBIN_RADIUS,
        "hemoglobin#bound" : HEMOGLOBIN_RADIUS,
        "hemoglobin#bound_1" : HEMOGLOBIN_RADIUS,
        "hemoglobin#bound_2" : HEMOGLOBIN_RADIUS,
        "oxygen" : OXYGEN_RADIUS,
        "oxygen#bound" : OXYGEN_RADIUS,
    }
    for name, radius in particles.items():
        diffCoeff = ReaddyUtil.calculate_diffusionCoefficient(
            radius, viscosity, TEMPERATURE
        )  # nm^2/s
        system.add_topology_species(name, diffCoeff)
    for name, radius in particles.items():
        system.potentials.add_box(
            particle_type=name,
            force_constant=FORCE_CONSTANT,
            origin=-0.5 * box_potential_size,
            extent=box_potential_size,
        )
        for other_name, other_radius in particles.items():
            bond_force = FORCE_CONSTANT
            if "oxygen" in name or "oxygen" in other_name:
                bond_force *= 0.01
            system.topologies.configure_harmonic_bond(
                name, other_name, bond_force, radius + other_radius
            )
            system.potentials.add_harmonic_repulsion(
                name, other_name, FORCE_CONSTANT, radius + other_radius
            )
    hb_types = [
        "hemoglobin", "hemoglobin#1", "hemoglobin#2", 
        "hemoglobin#bound", "hemoglobin#bound_1", "hemoglobin#bound_2",
    ]
    ReaddyUtil().add_angle(
        hb_types, hb_types, hb_types, 
        FORCE_CONSTANT, HEMOGLOBIN_ANGLE, system
    )

def reaction_function_finish_bind(topology):
    global n_binding_events
    n_binding_events += 1
    recipe = readdy.StructuralReactionRecipe(topology)
    v_oxygen = ReaddyUtil.get_first_vertex_of_types(
        topology,
        ["oxygen#bound"],
        error_msg="Failed to find bound oxygen while associating",
    )
    recipe.separate_vertex(v_oxygen)
    recipe.change_particle_type(v_oxygen, "oxygen#remove")
    recipe.change_topology_type("Hemoglobin")
    return recipe
    
def reaction_function_start_unbind(topology):
    recipe = readdy.StructuralReactionRecipe(topology)
    v_hemoglobin = ReaddyUtil.get_first_vertex_of_types(
        topology,
        ["hemoglobin#bound", "hemoglobin#bound_1", "hemoglobin#bound_2"],
    )
    if v_hemoglobin is None:
        return recipe
    global n_unbinding_events
    n_unbinding_events += 1
    pos = ReaddyUtil.get_vertex_position(topology, v_hemoglobin)
    recipe.append_particle(
        [v_hemoglobin],
        "oxygen#bound",
        pos + np.array([0, 0, HEMOGLOBIN_RADIUS + OXYGEN_RADIUS]),
    )
    recipe.change_topology_type("Dissociating1")
    return recipe

def reaction_function_finish_unbind(topology):
    recipe = readdy.StructuralReactionRecipe(topology)
    v_oxygen = ReaddyUtil.get_first_vertex_of_types(
        topology,
        ["oxygen#bound"],
        error_msg="Failed to find bound oxygen while dissociating",
    )
    v_hemoglobin = ReaddyUtil.get_neighbor_of_types(
        topology,
        v_oxygen,
        ["hemoglobin#bound", "hemoglobin#bound_1", "hemoglobin#bound_2"],
        [],
        error_msg="Failed to find bound hemoglobin neighbor of bound oxygen",
    )
    recipe.remove_edge(v_hemoglobin, v_oxygen)
    pt = topology.particle_type_of_vertex(v_hemoglobin)
    if pt[-1] == "1" or pt[-1] == "2":
        new_pt = f"hemoglobin#{pt[-1]}"
    else:
        new_pt = "hemoglobin"
    recipe.change_particle_type(v_hemoglobin, new_pt)
    recipe.change_particle_type(v_oxygen, "oxygen")
    recipe.change_topology_type("Dissociating2")
    return recipe

def reaction_function_cleanup(topology):
    recipe = readdy.StructuralReactionRecipe(topology)
    if len(topology.graph.get_vertices()) > 1:
        recipe.change_topology_type("Hemoglobin")
    else:
        recipe.change_topology_type("Oxygen")
    return recipe

def add_reactions(system):
    system.topologies.add_spatial_reaction(
        "Bind: Hemoglobin(hemoglobin) + Oxygen(oxygen) -> Associating(hemoglobin#bound--oxygen#bound)",
        rate=O2_BINDING_RATE,
        radius=HEMOGLOBIN_RADIUS + OXYGEN_RADIUS,
    )
    system.topologies.add_spatial_reaction(
        "Bind1: Hemoglobin(hemoglobin#1) + Oxygen(oxygen) -> Associating(hemoglobin#bound_1--oxygen#bound)",
        rate=O2_BINDING_RATE,
        radius=HEMOGLOBIN_RADIUS + OXYGEN_RADIUS,
    )
    system.topologies.add_spatial_reaction(
        "Bind2: Hemoglobin(hemoglobin#2) + Oxygen(oxygen) -> Associating(hemoglobin#bound_2--oxygen#bound)",
        rate=O2_BINDING_RATE,
        radius=HEMOGLOBIN_RADIUS + OXYGEN_RADIUS,
    )
    system.topologies.add_structural_reaction(
        "Finish_Bind",
        topology_type="Associating",
        reaction_function=reaction_function_finish_bind,
        rate_function=ReaddyUtil.rate_function_infinity,
    )
    system.reactions.add("Cleanup_Oxygen: oxygen#remove ->", rate=1e30)
    system.topologies.add_structural_reaction(
        "Unbind1",
        topology_type="Hemoglobin",
        reaction_function=reaction_function_start_unbind,
        rate_function=lambda x: O2_UNBINDING_RATE,
    )
    system.topologies.add_structural_reaction(
        "Unbind2",
        topology_type="Dissociating1",
        reaction_function=reaction_function_finish_unbind,
        rate_function=ReaddyUtil.rate_function_infinity,
    )
    system.topologies.add_structural_reaction(
        "Cleanup",
        topology_type="Dissociating2",
        reaction_function=reaction_function_cleanup,
        rate_function=ReaddyUtil.rate_function_infinity,
    )

def create_system():
    custom_units = {
        "length_unit": "nanometer",
        "time_unit": "nanosecond",
    }
    system = readdy.ReactionDiffusionSystem(
        box_size=3 * [BOX_SIZE],
        periodic_boundary_conditions=3 * [False],
        unit_system=custom_units,
    )
    system.temperature = TEMPERATURE
    add_types_and_potentials(system)
    add_reactions(system)
    return system

def hemoglobin_lattice_positions():
    result = []
    side_length = 3. * HEMOGLOBIN_RADIUS
    offset = ((N_HEMOGLOBIN_SIDE - 1) / 2.) * side_length
    n_hemoglobin = N_HEMOGLOBIN_SIDE ** 3
    for ix in range(n_hemoglobin):
        result.append([
            (math.floor(ix / (N_HEMOGLOBIN_SIDE ** 2)) % N_HEMOGLOBIN_SIDE) * side_length - offset, 
            (math.floor(ix / float(N_HEMOGLOBIN_SIDE)) % N_HEMOGLOBIN_SIDE) * side_length - offset, 
            (ix % N_HEMOGLOBIN_SIDE) * side_length - offset,
        ])
    return np.array(result)

def hemoglobin_particle_states():
    result = []
    for ix in range(4):
        number = str(ix + 1) if ix < 2 else ""
        if random.random() > START_PERCENT_O2_BOUND / 100.:
            delimiter = "#" if number else ""
            result.append(f"hemoglobin{delimiter}{number}")
        else:
            delimiter = "_" if number else ""
            result.append(f"hemoglobin#bound{delimiter}{number}")
    return result

def config_init_conditions(simulation):
    hemoglobin_positions = hemoglobin_lattice_positions()
    r = 1.5 * HEMOGLOBIN_RADIUS
    subunit_positions = np.array([
        [-r, 0, 0],
        [r, 0, 0],
        [0, -r, 0],
        [0, r, 0],
    ])
    for center_position in hemoglobin_positions:
        states = hemoglobin_particle_states()
        top = simulation.add_topology(
            "Hemoglobin", states, center_position + subunit_positions
        )
        for x in range(4):
            for y in range(x + 1, 4, 1):
                top.get_graph().add_edge(x, y)
    oxygen_positions = (np.random.uniform(size=(N_OXYGEN, 3)) - 0.5) * BOX_SIZE
    for position in oxygen_positions:
        simulation.add_topology(
            "Oxygen", ["oxygen"], np.array([position])
        )
        
def correct_rotation(time_ix, raw_rot, prev_raw_rot, diff_rot):
    if time_ix < 1:
        return raw_rot, Rotation.from_rotvec(np.zeros(3))
    curr_diff = raw_rot * prev_raw_rot.inv()
    new_diff = diff_rot * curr_diff.inv()
    curr_rot = raw_rot * new_diff
    return curr_rot, new_diff
        
def add_unified_hemoglobin(agent_data):
    total_steps = agent_data.times.shape[0]
    n_hemoglobin = N_HEMOGLOBIN_SIDE ** 3
    start_ix = agent_data.get_dimensions().max_agents
    end_ix = start_ix + n_hemoglobin
    new_agent_data = agent_data.get_copy_with_increased_buffer_size(
        DimensionData(
            total_steps=total_steps, 
            max_agents=n_hemoglobin,
        )
    )
    new_agent_data.n_agents += n_hemoglobin
    new_agent_data.unique_ids[:,start_ix:end_ix] = np.array(range(start_ix, end_ix, 1))
    for time_ix in range(total_steps):
        new_agent_data.types[time_ix] += n_hemoglobin * ["hemoglobin#unified"]
    n_subunits = 4 * n_hemoglobin
    positions = np.zeros((total_steps, n_hemoglobin, 3))
    rotations = np.zeros((total_steps, n_hemoglobin, 3))
    prev_rotations = n_hemoglobin * [None]
    diff_rotations = n_hemoglobin * [None]
    for time_ix in range(total_steps):
        for sub_ix in range(0, n_subunits, 4):
            hb_ix = math.floor(sub_ix / 4.)
            sub_positions = agent_data.positions[time_ix][sub_ix:sub_ix + 4]
            center_pos = np.mean(sub_positions, axis=0)
            positions[time_ix][hb_ix] = center_pos
            positions[time_ix][hb_ix + 1] = center_pos + np.array([5, 0, 0])
            raw_rotation = Rotation.from_matrix(
                ReaddyUtil.get_orientation_from_positions([
                    sub_positions[0], center_pos, sub_positions[1]
                ])
            )
            corrected_rotation, diff_rotations[hb_ix] = correct_rotation(
                time_ix, raw_rotation, prev_rotations[hb_ix], diff_rotations[hb_ix]
            )
            rotations[time_ix][hb_ix] = corrected_rotation.as_euler("XYZ", degrees=False)
            prev_rotations[hb_ix] = copy.copy(raw_rotation)
    new_agent_data.positions[:,start_ix:end_ix] = positions
    new_agent_data.radii[:,start_ix:end_ix] = np.ones(n_hemoglobin)
    new_agent_data.rotations[:,start_ix:end_ix] = rotations
    new_agent_data.display_data["hemoglobin#unified"] = DisplayData(
        name="hemoglobin#unified",
        display_type=DISPLAY_TYPE.OBJ,
        url="https://aics-simularium-data.s3.us-east-2.amazonaws.com/meshes/obj/maya_gizmo.obj",
        radius=1.0,
        color="#666666",
    )
    return new_agent_data
        
def visualize(output_name, total_steps):
    oxygen_display_data = DisplayData(
        name="oxygen",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=OXYGEN_RADIUS,
        color="#ffffff",
    )
    hb_display_data = DisplayData(
        name="hemoglobin#deoxy",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=HEMOGLOBIN_RADIUS,
        color="#0000ff",
    )
    hb_display_data1 = copy.copy(hb_display_data)
    hb_display_data1.name = "hemoglobin#deoxy_1"
    hb_display_data2 = copy.copy(hb_display_data)
    hb_display_data2.name = "hemoglobin#deoxy_2"
    hb_bound_display_data = DisplayData(
        name="hemoglobin#oxy",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=HEMOGLOBIN_RADIUS,
        color="#660000",
    )
    hb_bound_display_data1 = copy.copy(hb_bound_display_data)
    hb_bound_display_data1.name = "hemoglobin#oxy_1"
    hb_bound_display_data2 = copy.copy(hb_bound_display_data)
    hb_bound_display_data2.name = "hemoglobin#oxy_2"
    traj_data = ReaddyConverter(ReaddyData(
        timestep=TIMESTEP * max(int(total_steps / 1000.0), 1) * 1e-3,
        path_to_readdy_h5=output_name + ".h5",
        meta_data=MetaData(
            box_size=np.array(3 * [BOX_SIZE]),
            scale_factor=1.0,
        ),
        display_data={
            "hemoglobin": hb_display_data,
            "hemoglobin#1": hb_display_data1,
            "hemoglobin#2": hb_display_data2,
            "hemoglobin#bound": hb_bound_display_data,
            "hemoglobin#bound_1": hb_bound_display_data1,
            "hemoglobin#bound_2": hb_bound_display_data2,
            "oxygen": oxygen_display_data,
            "oxygen#bound": oxygen_display_data,
            "oxygen#remove": oxygen_display_data,
        },
        time_units=UnitData("ms"),
        spatial_units=UnitData("nm"),
    ))._data
    traj_data.agent_data = add_unified_hemoglobin(traj_data.agent_data)
    TrajectoryConverter(traj_data).save(output_name)


def main():
    args = parse_args()
    system = create_system()
    total_steps = int(float(args.total_steps))
    simulation = ReaddyUtil.create_readdy_simulation(
        system,
        n_cpu=4,
        sim_name=args.output_name,
        total_steps=total_steps,
        record=True,
    )
    config_init_conditions(simulation)
    simulation.run(
        n_steps=total_steps,
        timestep=TIMESTEP,
        show_summary=False,
    )
    visualize(args.output_name, total_steps)
    print(f"{n_binding_events} binding events.")
    print(f"{n_unbinding_events} unbinding events.")


if __name__ == "__main__":
    main()
