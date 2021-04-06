#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import readdy
import random

from ..common import ReaddyUtil
from .actin_generator import ActinGenerator
from .actin_structure import ActinStructure
from .fiber_data import FiberData
from .arp_data import ArpData
from .curve_point_data import CurvePointData


verbose = False
def set_verbose(x):
    """
    print debug statements?
    """
    global verbose
    verbose = x
    return x

box_size = 0.
def set_box_size(x):
    """
    set the box_size
    """
    global box_size
    box_size = x
    return x

rates = {}
def set_rates(r):
    """
    set reaction rates
    """
    global rates
    rates = r

def get_new_arp23(topology):
    """
    get a new arp3 and its unbranched arp2 neighbor,
    meaning the arp2/3 dimer has just bound
    """
    for vertex in topology.graph.get_vertices():
        pt = topology.particle_type_of_vertex(vertex)
        if "arp3#new" in pt:
            for neighbor in vertex:
                if topology.particle_type_of_vertex(neighbor.get()) == "arp2":
                    return neighbor.get(), vertex
    return None, None

def cancel_branch_reaction(topology, recipe, actin_arp3, arp3):
    """
    Undo the branching spatial reaction if the structural reaction fails
    """
    if verbose:
        print("Canceling branch reaction")
    pt = topology.particle_type_of_vertex(actin_arp3)
    recipe.remove_edge(actin_arp3, arp3)
    ReaddyUtil.set_flags(topology, recipe, arp3, [], ["new"], True)
    recipe.change_topology_type(
        "Actin-Polymer#Fail-Branch-{}".format("ATP" if ("ATP" in pt) else "ADP"))

def get_actin_number(topology, vertex, offset):
    """
    get the type number for an actin plus the given offset in range [-1, 1]
    (i.e. return 3 for type = "actin#ATP_1" and offset = -1)
    """
    pt = topology.particle_type_of_vertex(vertex)
    if not "actin" in pt:
        raise Exception(f"Failed to get actin number: {pt} is not actin\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    return ReaddyUtil.calculate_polymer_number(int(pt[-1]), offset)

def get_all_polymer_actin_types(vertex_type):
    """
    get a list of all numbered versions of a type
    (e.g. for "actin#ATP" return 
    ["actin#ATP_1", "actin#ATP_2", "actin#ATP_3"])
    """
    spacer = "_"
    if not "#" in vertex_type:
        spacer = "#"
    return ["{}{}1".format(vertex_type, spacer),
            "{}{}2".format(vertex_type, spacer),
            "{}{}3".format(vertex_type, spacer)]

def get_actin_rotation(positions):
    """
    get the difference in the actin's current orientation
    compared to the initial orientation as a rotation matrix
    positions = [previous actin position, middle actin position, next actin position]
    """
    positions[0] = ReaddyUtil.get_non_periodic_boundary_position(
        positions[1], positions[0], box_size)
    positions[2] = ReaddyUtil.get_non_periodic_boundary_position(
        positions[1], positions[2], box_size)
    current_orientation = ReaddyUtil.get_orientation_from_positions(positions)
    return np.matmul(current_orientation, np.linalg.inv(ActinStructure.orientation()))

def get_position_for_new_vertex(neighbor_positions, offset_vector):
    """
    get the offset vector in the local space for the actin at neighbor_positions[1]
    neighbor_positions = [
        previous actin position,
        middle actin position,
        next actin position
    ]
    """
    rotation = get_actin_rotation(neighbor_positions)
    if rotation is None:
        return None
    vector_to_new_pos = np.squeeze(np.array(np.dot(rotation, offset_vector)))
    return (neighbor_positions[1] + vector_to_new_pos).tolist()

def get_prev_branch_actin(topology, vertex, last_vertex_id, max_edges):
    """
    recurse up the chain until first branch actin is found or max_edges is reached
    """
    for neighbor in vertex:
        n_id = topology.particle_id_of_vertex(neighbor)
        if n_id == last_vertex_id:
            continue
        pt = topology.particle_type_of_vertex(neighbor)
        if "branch_" in pt:
            return neighbor.get(), max_edges
        else:
            if max_edges <= 1:
                return None, max_edges
            return get_prev_branch_actin(topology, neighbor.get(), n_id, max_edges-1)
    return None, max_edges

def get_branch_orientation_vertices_and_offset(topology, vertex):
    """
    get orientation vertices [actin, actin_arp2, actin_arp3]
    for a new actin within 3 actins of a branch,
    as well as the offset vector
    """
    v_arp2 = ReaddyUtil.get_neighbor_of_types(
        topology, vertex, ["arp2", "arp2#branched"], [])
    offset_index = 0
    if v_arp2 is None:
        v_branch, edges = get_prev_branch_actin(topology, vertex, None, 3)
        if v_branch is None:
            raise Exception(f"Failed to set position: couldn't find arp2 or first branch actin\n"
                f"{ReaddyUtil.topology_to_string(topology)}")
        offset_index = 3 - edges
        v_arp2 = ReaddyUtil.get_neighbor_of_types(
            topology, v_branch, ["arp2", "arp2#branched"], [])
        if v_arp2 is None:
            raise Exception(f"Failed to set position: couldn't find arp2\n"
                f"{ReaddyUtil.topology_to_string(topology)}")
    v_arp3 = ReaddyUtil.get_neighbor_of_types(
        topology, v_arp2, ["arp3", "arp3#ATP", "arp3#new", "arp3#new_ATP"], [])
    if v_arp3 is None:
        raise Exception(f"Failed to set position: couldn't find arp3\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    actin_types = (get_all_polymer_actin_types("actin")
        + get_all_polymer_actin_types("actin#ATP")
        + get_all_polymer_actin_types("actin#barbed")
        + get_all_polymer_actin_types("actin#barbed_ATP"))
    v_actin_arp3 = ReaddyUtil.get_neighbor_of_types(
        topology, v_arp3, actin_types, [])
    if v_actin_arp3 is None:
        raise Exception(f"Failed to set position: couldn't find actin_arp3\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    n_pointed = get_actin_number(topology, v_actin_arp3, -1)
    actin_types = ["actin#ATP_{}".format(n_pointed), "actin#{}".format(n_pointed)]
    v_actin_arp2 = ReaddyUtil.get_neighbor_of_types(
        topology, v_actin_arp3, actin_types, [])
    if v_actin_arp2 is None:
        raise Exception(f"Failed to set position: couldn't find actin_arp2\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    n_pointed = get_actin_number(topology, v_actin_arp2, -1)
    actin_types = [
        "actin#ATP_{}".format(n_pointed), "actin#{}".format(n_pointed),
        "actin#pointed_ATP_{}".format(n_pointed), "actin#pointed_{}".format(n_pointed)
    ]
    if n_pointed == 1:
        actin_types += ["actin#branch_1", "actin#branch_ATP_1"]
    v_prev = ReaddyUtil.get_neighbor_of_types(
        topology, v_actin_arp2, actin_types, [v_actin_arp3])
    if v_prev is None:
        raise Exception("Failed to set position: couldn't find v_prev, "
            f"actin_arp2 is {ReaddyUtil.vertex_to_string(topology, v_actin_arp2)} "
            f"and actin_arp3 is {ReaddyUtil.vertex_to_string(topology, v_actin_arp3)}\n"
            + str(ReaddyUtil.topology_to_string(topology)))
    return ([v_prev, v_actin_arp2, v_actin_arp3],
        ActinStructure.mother1_to_branch_actin_vectors()[offset_index])

def set_end_vertex_position(topology, recipe, v_new, barbed):
    """
    set the position of a new pointed or barbed vertex
    """
    vertices = []
    offset_vector = (ActinStructure.mother1_to_mother3_vector() 
                     if barbed else ActinStructure.mother1_to_mother_vector())
    at_branch = False
    vertices.append(ReaddyUtil.get_neighbor_of_type(
        topology, v_new, "actin", False))
    if vertices[0] is None:
        vertices, offset_vector = get_branch_orientation_vertices_and_offset(
            topology, v_new)
        at_branch = True
    else:
        vertices.append(ReaddyUtil.get_neighbor_of_type(
            topology, vertices[0], "actin", False, [v_new]))
        if vertices[1] is None:
            vertices, offset_vector = get_branch_orientation_vertices_and_offset(
                topology, v_new)
            at_branch = True
        else:
            vertices.append(ReaddyUtil.get_neighbor_of_type(
                topology, vertices[1], "actin", False, [vertices[0]]))
            if vertices[2] is None:
                vertices, offset_vector = get_branch_orientation_vertices_and_offset(
                    topology, v_new)
                at_branch = True
    positions = []
    for v in vertices:
        positions.append(ReaddyUtil.get_vertex_position(topology, v))
    if barbed and not at_branch:
        positions = positions[::-1]
    pos = get_position_for_new_vertex(positions, offset_vector)
    if pos is None:
        raise Exception(f"Failed to set position: couldn't calculate position\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    recipe.change_particle_position(v_new, pos)

def set_new_trimer_vertex_position(
    topology, recipe, v_new, v_pointed, v_barbed
):
    """
    set the position of an actin monomer just added to a dimer to create a trimer
    """
    pos_new = ReaddyUtil.get_vertex_position(topology, v_new)
    pos_pointed = ReaddyUtil.get_vertex_position(topology, v_pointed)
    pos_barbed = ReaddyUtil.get_vertex_position(topology, v_barbed)
    v_barbed_to_pointed = pos_pointed - pos_barbed
    v_barbed_to_new = pos_new - pos_barbed
    current_angle = ReaddyUtil.get_angle_between_vectors(
        v_barbed_to_pointed, v_barbed_to_new)
    angle = ActinStructure.actin_to_actin_angle() - current_angle
    axis = np.cross(v_barbed_to_pointed, v_barbed_to_new)
    pos = pos_barbed + ReaddyUtil.rotate(
        ActinStructure.actin_to_actin_distance() * ReaddyUtil.normalize(v_barbed_to_new), 
        axis, angle
    )
    recipe.change_particle_position(v_new, pos)

def set_arp23_vertex_position(
    topology, recipe, v_arp2, v_arp3, v_actin_arp2, v_actin_arp3
):
    """
    set the position of new arp2/3 vertices
    """
    actin_types = (get_all_polymer_actin_types("actin")
        + get_all_polymer_actin_types("actin#ATP")
        + get_all_polymer_actin_types("actin#pointed")
        + get_all_polymer_actin_types("actin#pointed_ATP")
        + ["actin#branch_1", "actin#branch_ATP_1"])
    v1 = ReaddyUtil.get_neighbor_of_types(
        topology, v_actin_arp2, actin_types, [v_actin_arp3])
    if v1 is None:
        raise Exception(f"Failed to set position: couldn't find v1\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    pos1 = ReaddyUtil.get_vertex_position(topology, v1)
    pos2 = ReaddyUtil.get_vertex_position(topology, v_actin_arp2)
    pos3 = ReaddyUtil.get_vertex_position(topology, v_actin_arp3)
    pos_arp2 = get_position_for_new_vertex(
        [pos1, pos2, pos3], ActinStructure.mother1_to_arp2_vector())
    if pos_arp2 is None:
        raise Exception(f"Failed to set position of arp2: couldn't calculate position\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    recipe.change_particle_position(v_arp2, pos_arp2)
    pos_arp3 = get_position_for_new_vertex(
        [pos1, pos2, pos3], ActinStructure.mother1_to_arp3_vector())
    if pos_arp3 is None:
        raise Exception(f"Failed to set position of arp3: couldn't calculate position\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    recipe.change_particle_position(v_arp3, pos_arp3)

def get_random_arp2(topology, with_ATP, with_branch):
    """
    get a random bound arp2 with the given arp3 nucleotide state
    and with or without a branch attached to the arp2
    """
    v_arp3s = ReaddyUtil.get_vertices_of_type(
        topology, "arp3#ATP" if with_ATP else "arp3", True)
    if len(v_arp3s) < 1:
        if verbose:
            print("Couldn't find arp3 with {}".format(
                "ATP" if with_ATP else "ADP"))
        return None
    v_arp2s = []
    for v_arp3 in v_arp3s:
        v_arp2 = ReaddyUtil.get_neighbor_of_types(
            topology, v_arp3, ["arp2#branched" if with_branch else "arp2"], [])
        if v_arp2 is not None:
            v_arp2s.append(v_arp2)
    if len(v_arp2s) < 1:
        if verbose:
            print("Couldn't find arp2 with{} branch".format(
                "out" if not with_branch else ""))
        return None
    return random.choice(v_arp2s)

def add_linear_fibers(simulation, n_fibers):
    """
    add linear fibers
    """
    length = 20
    positions = np.random.uniform(size=(n_fibers,3)) * box_size - box_size * 0.5
    for fiber in range(n_fibers):
        direction = ReaddyUtil.get_random_unit_vector()
        monomers = ActinGenerator.get_monomers({
            0: FiberData(0, [
                CurvePointData(
                    positions[fiber],
                    direction, 
                    np.array([1, 0, 0]), 
                    0,
                ),
                CurvePointData(
                    positions[fiber] + length * direction,
                    direction, 
                    np.array([1, 0, 0]), 
                    length,
                ),
            ])
        })
        top = simulation.add_topology("Actin-Polymer", monomers[0][1], np.array(monomers[0][0]))
        for e in monomers[0][2]:
            top.get_graph().add_edge(e[0], e[1])

def add_branched_fiber(simulation):
    """
    add a branched fiber
    """
    mother_fiber = FiberData(0, [
        CurvePointData(
            np.array([-50, 0, 0]), 
            np.array([1, 0, 0]), 
            np.array([0, 1, 0]),
            0,
        ),
        CurvePointData(
            np.array([50, 0, 0]), 
            np.array([1, 0, 0]), 
            np.array([0, 1, 0]),
            100,
        ),
    ])
    daughter_fiber = FiberData(1, [
        CurvePointData(
            np.array([0, 0, 0]), 
            np.array([0.31037692, 0.94885799, -0.05774669]), 
            np.array([0, 1, 0]),
            0,
        ),
        CurvePointData(
            0.5 * np.array([31.037692, 94.885799, -5.774669]), 
            np.array([0.31037692, 0.94885799, -0.05774669]), 
            np.array([0, 1, 0]),
            50,
        ),
    ])
    arp = ArpData(0, np.array([0, 0, 0]), True, False)
    arp.daughter_fiber = daughter_fiber
    mother_fiber.nucleated_arps.append(arp)
    monomers = ActinGenerator.get_monomers({
        0: mother_fiber
    })
    top = simulation.add_topology("Actin-Polymer", monomers[0][1], np.array(monomers[0][0]))
    for e in monomers[0][2]:
        top.get_graph().add_edge(e[0], e[1])

def add_dimer(position, simulation):
    """
    add an actin dimer fiber
    """
    positions = np.array(
        [[0, 0, 0], ActinStructure.actin_to_actin_distance() * ReaddyUtil.get_random_unit_vector()])
    types = ["actin#pointed_ATP_1", "actin#barbed_ATP_2"]
    top = simulation.add_topology("Actin-Dimer", types, position + positions)
    top.get_graph().add_edge(0, 1)

def add_dimers(n, box_size, simulation):
    """
    add actin dimers
    """
    positions = np.random.uniform(size=(n,3)) * box_size - box_size * 0.5
    for p in range(len(positions)):
        add_dimer(positions[p], simulation)

def add_monomers(n, box_size, simulation):
    """
    add free actin
    """
    positions = np.random.uniform(size=(n,3)) * box_size - box_size * 0.5
    for p in range(len(positions)):
        simulation.add_topology(
            "Actin-Monomer", ["actin#free_ATP"], np.array([positions[p]]))

def add_arp23_dimers(n, box_size, simulation):
    """
    add arp2/3 dimers
    """
    positions = np.random.uniform(size=(n,3)) * box_size - box_size * 0.5
    for p in range(len(positions)):
        top = simulation.add_topology("Arp23-Dimer", ["arp2", "arp3#ATP"],
            np.array([positions[p], positions[p]
            + 4. * ReaddyUtil.get_random_unit_vector()]))
        top.get_graph().add_edge(0, 1)

def add_capping_protein(n, box_size, simulation):
    """
    add free capping protein
    """
    positions = np.random.uniform(size=(n,3)) * box_size - box_size * 0.5
    for p in range(len(positions)):
        simulation.add_topology("Cap", ["cap"], np.array([positions[p]]))

def reaction_function_reverse_dimerize(topology):
    """
    reaction function for a dimer falling apart
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Reverse Dimerize")
    v_barbed = ReaddyUtil.get_first_vertex_of_types(topology,
        ["actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3",
        "actin#barbed_1", "actin#barbed_2", "actin#barbed_3"])
    if v_barbed is None:
        raise Exception(f"Failed to find barbed end of dimer\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    v_pointed = ReaddyUtil.get_first_neighbor(topology, v_barbed, [])
    if v_pointed is None:
        raise Exception(f"Failed to find pointed end of dimer\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    recipe.remove_edge(v_barbed, v_pointed)
    recipe.change_particle_type(v_barbed, "actin#free_ATP")
    recipe.change_particle_type(v_pointed, "actin#free_ATP")
    recipe.change_topology_type("Actin-Monomer")
    return recipe

def reaction_function_finish_trimerize(topology):
    """
    reaction function for a trimer forming
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Trimerize")
    v_new = ReaddyUtil.get_first_vertex_of_types(topology,
        ["actin#new", "actin#new_ATP"])
    if v_new is None:
        raise Exception(f"Failed to find new vertex in trimer\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    v_neighbor1 = ReaddyUtil.get_first_neighbor(topology, v_new, [])
    if v_neighbor1 is None:
        raise Exception(f"Failed to find first neighbor of new vertex in trimer\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    v_neighbor2 = ReaddyUtil.get_first_neighbor(topology, v_neighbor1, [v_new])
    if v_neighbor2 is None:
        raise Exception(f"Failed to find second neighbor of new vertex in trimer\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    ReaddyUtil.set_flags(topology, recipe, v_new,
        ["barbed", str(get_actin_number(topology, v_neighbor1, 1))],
        ["new"], True)
    set_new_trimer_vertex_position(
        topology, recipe, v_new, v_neighbor2, v_neighbor1)
    recipe.change_topology_type("Actin-Trimer")
    return recipe

def reaction_function_reverse_trimerize(topology):
    """
    reaction function for removing ATP-actin from a trimer
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Reverse Trimerize")
    v_barbed = ReaddyUtil.get_first_vertex_of_types(topology,
        ["actin#barbed_ATP_1", "actin#barbed_ATP_2", "actin#barbed_ATP_3",
        "actin#barbed_1", "actin#barbed_2", "actin#barbed_3"])
    if v_barbed is None:
        raise Exception(f"Failed to find barbed end in trimer\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    v_neighbor = ReaddyUtil.get_first_neighbor(topology, v_barbed, [])
    if v_neighbor is None:
        raise Exception(f"Failed to find neighbor of barbed end in trimer\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    recipe.remove_edge(v_barbed, v_neighbor)
    recipe.change_particle_type(v_barbed, "actin#free_ATP")
    ReaddyUtil.set_flags(topology, recipe, v_neighbor, ["barbed"], [], True)
    recipe.change_topology_type("Actin-Polymer#Shrinking")
    return recipe

def reaction_function_finish_pointed_grow(topology):
    """
    reaction function for the pointed end growing
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Grow Pointed")
    v_new = ReaddyUtil.get_first_vertex_of_types(topology,
        ["actin#new", "actin#new_ATP"])
    if v_new is None:
        raise Exception(f"Failed to find new vertex at pointed end\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    v_neighbor = ReaddyUtil.get_first_neighbor(topology, v_new, [])
    if v_neighbor is None:
        raise Exception(f"Failed to find neighbor of new pointed end\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    ReaddyUtil.set_flags(topology, recipe, v_new,
        ["pointed", str(get_actin_number(topology, v_neighbor, -1))],
        ["new"], True)
    recipe.change_topology_type("Actin-Polymer")
    set_end_vertex_position(topology, recipe, v_new, False)
    return recipe

def reaction_function_finish_barbed_grow(topology):
    """
    reaction function for the barbed end growing
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Grow Barbed")
    v_new = ReaddyUtil.get_first_vertex_of_types(topology,
        ["actin#new", "actin#new_ATP"])
    if v_new is None:
        raise Exception(f"Failed to find new barbed end\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    v_neighbor = ReaddyUtil.get_first_neighbor(topology, v_new, [])
    if v_neighbor is None:
        raise Exception(f"Failed to find neighbor of new barbed end\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    ReaddyUtil.set_flags(topology, recipe, v_new,
        ["barbed", str(get_actin_number(topology, v_neighbor, 1))],
        ["new"], True)
    set_end_vertex_position(topology, recipe, v_new, True)
    recipe.change_topology_type("Actin-Polymer")
    return recipe

def reaction_function_finish_arp_bind(topology):
    """
    reaction function to finish a branching reaction
    (triggered by a spatial reaction)
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Bind Arp2/3")
    v_arp2, v_arp3 = get_new_arp23(topology)
    if v_arp2 is None or v_arp3 is None:
        raise Exception(f"Failed to find new arp2 and arp3\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    v_actin_barbed = ReaddyUtil.get_first_neighbor(topology, v_arp3, [v_arp2])
    if v_actin_barbed is None:
        raise Exception(f"Failed to find new actin_arp3\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    # make sure arp3 binds to the barbed end neighbor of the actin bound to arp2
    n_pointed = get_actin_number(topology, v_actin_barbed, -1)
    actin_types = ["actin#ATP_{}".format(n_pointed), "actin#{}".format(n_pointed),
         "actin#pointed_ATP_{}".format(n_pointed), "actin#pointed_{}".format(n_pointed)]
    if n_pointed == 1:
        actin_types += ["actin#branch_1", "actin#branch_ATP_1"]
    v_actin_pointed = ReaddyUtil.get_neighbor_of_types(
        topology, v_actin_barbed, actin_types, [])
    if v_actin_pointed is not None:
        pointed_type = topology.particle_type_of_vertex(v_actin_pointed)
        if "pointed" in pointed_type or "branch" in pointed_type:
            if verbose:
                print(f"Branch is starting exactly at a pointed end")
            cancel_branch_reaction(
                topology, recipe, v_actin_barbed, v_arp3)
            return recipe
    else:
        if verbose:
            print(f"Couldn't find actin_arp2 with number {n_pointed}")
        cancel_branch_reaction(
            topology, recipe, v_actin_barbed, v_arp3)
        return recipe
    ReaddyUtil.set_flags(topology, recipe, v_arp3, [], ["new"], True)
    recipe.add_edge(v_actin_pointed, v_arp2)
    recipe.change_topology_type("Actin-Polymer")
    set_arp23_vertex_position(
        topology, recipe, v_arp2, v_arp3, v_actin_pointed, v_actin_barbed)
    return recipe

def reaction_function_finish_start_branch(topology):
    """
    reaction function for adding the first actin to an arp2/3 to start a branch
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Start Branch")
    v_new = ReaddyUtil.get_first_vertex_of_types(topology,
        ["actin#new", "actin#new_ATP"])
    if v_new is None:
        raise Exception(f"Failed to find new branch actin\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    ReaddyUtil.set_flags(
        topology, recipe, v_new, ["barbed", "1", "branch"], ["new"], True)
    recipe.change_topology_type("Actin-Polymer")
    set_end_vertex_position(topology, recipe, v_new, True)
    return recipe

def do_shrink(topology, recipe, barbed, ATP):
    """
    remove an (ATP or ADP)-actin from the (barbed or pointed) end
    """
    end_type = "actin#{}{}".format(
        "barbed" if barbed else "pointed", "_ATP" if ATP else "")
    v_end = ReaddyUtil.get_random_vertex_of_types(
        topology, get_all_polymer_actin_types(end_type))
    if v_end is None:
        if verbose:
            print("Couldn't find end actin to remove")
        return False
    v_arp = ReaddyUtil.get_neighbor_of_types(
        topology, v_end, ["arp3", "arp3#ATP", "arp2", "arp2#branched"], [])
    if v_arp is not None:
        if verbose:
            print("Couldn't remove actin because a branch was attached")
        return False
    v_neighbor = ReaddyUtil.get_neighbor_of_types(
        topology, v_end, 
        get_all_polymer_actin_types("actin")
        + get_all_polymer_actin_types("actin#ATP")
        + ["actin#branch_1", "actin#branch_ATP_1"], [])
    if v_neighbor is None:
        if verbose:
            print("Couldn't find plain actin neighbor of actin to remove")
        return False
    if not barbed:
        v_arp2 = ReaddyUtil.get_neighbor_of_types(
            topology, v_neighbor, ["arp2", "arp2#branched"], [])
        if v_arp2 is not None:
            if verbose:
                print("Couldn't remove actin because a branch was attached to its barbed neighbor")
            return False
    recipe.remove_edge(v_end, v_neighbor)
    recipe.change_particle_type(
        v_end, "actin#free" if not ATP else "actin#free_ATP")
    ReaddyUtil.set_flags(topology, recipe, v_neighbor,
        ["barbed"] if barbed else ["pointed"], [], True)
    recipe.change_topology_type("Actin-Polymer#Shrinking")
    return True

def reaction_function_pointed_shrink_ATP(topology):
    """
    reaction function to remove an ATP-actin from the pointed end
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Shrink pointed ATP")
    if not do_shrink(topology, recipe, False, True):
        recipe.change_topology_type("Actin-Polymer#Fail-Pointed-Shrink-ATP")
    return recipe

def reaction_function_pointed_shrink_ADP(topology):
    """
    reaction function to remove an ADP-actin from the pointed end
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Shrink pointed ADP")
    if not do_shrink(topology, recipe, False, False):
        recipe.change_topology_type("Actin-Polymer#Fail-Pointed-Shrink-ADP")
    return recipe

def reaction_function_barbed_shrink_ATP(topology):
    """
    reaction function to remove an ATP-actin from the barbed end
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Shrink barbed ATP")
    if not do_shrink(topology, recipe, True, True):
        recipe.change_topology_type("Actin-Polymer#Fail-Barbed-Shrink-ATP")
    return recipe

def reaction_function_barbed_shrink_ADP(topology):
    """
    reaction function to remove an ADP-actin from the barbed end
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Shrink barbed ADP")
    if not do_shrink(topology, recipe, True, False):
        recipe.change_topology_type("Actin-Polymer#Fail-Barbed-Shrink-ADP")
    return recipe

def reaction_function_cleanup_shrink(topology):
    """
    reaction function for finishing a reverse polymerization reaction
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Cleanup Shrink")
    new_type = ""
    if len(topology.graph.get_vertices()) < 2:
        v_cap = ReaddyUtil.get_vertex_of_type(topology, "cap", True)
        if v_cap is not None:
            new_type = "Cap"
        else:
            new_type = "Actin-Monomer"
    elif len(topology.graph.get_vertices()) < 3:
        v_arp2 = ReaddyUtil.get_vertex_of_type(topology, "arp2", True)
        if v_arp2 is not None:
            new_type = "Arp23-Dimer"
        else:
            new_type = "Actin-Dimer"
    elif len(topology.graph.get_vertices()) < 4:
        new_type = "Actin-Trimer"
    else:
        new_type = "Actin-Polymer"
    if verbose:
        print(f"cleaned up {new_type}")
    recipe.change_topology_type(new_type)
    return recipe

def reaction_function_hydrolyze_actin(topology):
    """
    reaction function to hydrolyze a filamentous ATP-actin to ADP-actin
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Hydrolyze Actin")
    v = ReaddyUtil.get_random_vertex_of_types(
        topology, get_all_polymer_actin_types("actin#ATP")
        + get_all_polymer_actin_types("actin#pointed_ATP")
        + get_all_polymer_actin_types("actin#barbed_ATP")
        + ["actin#branch_barbed_ATP_1", "actin#branch_ATP_1"])
    if v is None:
        if verbose:
            print("Couldn't find ATP-actin")
        recipe.change_topology_type("Actin-Polymer#Fail-Hydrolysis-Actin")
        return recipe
    ReaddyUtil.set_flags(topology, recipe, v, [], ["ATP"], True)
    return recipe

def reaction_function_hydrolyze_arp(topology):
    """
    reaction function to hydrolyze a arp2/3
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Hydrolyze Arp2/3")
    v = ReaddyUtil.get_random_vertex_of_types(topology, ["arp3#ATP"])
    if v is None:
        if verbose:
            print("Couldn't find ATP-arp3")
        recipe.change_topology_type("Actin-Polymer#Fail-Hydrolysis-Arp")
        return recipe
    ReaddyUtil.set_flags(topology, recipe, v, [], ["ATP"], True)
    return recipe

def reaction_function_nucleotide_exchange_actin(topology):
    """
    reaction function to exchange ATP for ADP in free actin
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Nucleotide Exchange Actin")
    v = ReaddyUtil.get_vertex_of_type(topology, "actin#free", True)
    if v is None:
        if verbose:
            print("Couldn't find ADP-actin")
        recipe.change_topology_type("Actin-Polymer#Fail-Nucleotide-Exchange-Actin")
        return recipe
    ReaddyUtil.set_flags(topology, recipe, v, ["ATP"], [], True)
    return recipe

def reaction_function_nucleotide_exchange_arp(topology):
    """
    reaction function to exchange ATP for ADP in free Arp2/3
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Nucleotide Exchange Arp2/3")
    v = ReaddyUtil.get_vertex_of_type(topology, "arp3", True)
    if v is None:
        if verbose:
            print("Couldn't find ADP-arp3")
        recipe.change_topology_type("Actin-Polymer#Fail-Nucleotide-Exchange-Arp")
        return recipe
    ReaddyUtil.set_flags(topology, recipe, v, ["ATP"], [], True)
    return recipe

def do_arp23_unbind(topology, recipe, with_ATP):
    """
    dissociate an arp2/3 from a mother filament
    """
    v_arp2 = get_random_arp2(topology, with_ATP, False)
    if v_arp2 is None:
        recipe.change_topology_type("Actin-Polymer#Fail-Arp-Unbind-{}".format(
            "ATP" if with_ATP else "ADP"))
        if verbose:
            print("Couldn't find unbranched {}-arp2".format("ATP" if with_ATP else "ADP"))
        return recipe
    actin_types = (get_all_polymer_actin_types("actin")
        + get_all_polymer_actin_types("actin#ATP")
        + get_all_polymer_actin_types("actin#pointed")
        + get_all_polymer_actin_types("actin#pointed_ATP")
        + get_all_polymer_actin_types("actin#barbed")
        + get_all_polymer_actin_types("actin#barbed_ATP")
        + ["actin#branch_1", "actin#branch_ATP_1"])
    v_actin_arp2 = ReaddyUtil.get_neighbor_of_types(
        topology, v_arp2, actin_types, [])
    if v_actin_arp2 is None:
        raise Exception(f"Failed to find actin_arp2\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    v_arp3 = ReaddyUtil.get_neighbor_of_types(
        topology, v_arp2, ["arp3", "arp3#ATP"], [])
    if v_arp3 is None:
        raise Exception(f"Failed to find arp3\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    v_actin_arp3 = ReaddyUtil.get_neighbor_of_types(
        topology, v_arp3, actin_types, [])
    if v_actin_arp3 is None:
        raise Exception(f"Failed to find actin_arp3\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    recipe.remove_edge(v_arp2, v_actin_arp2)
    recipe.remove_edge(v_arp3, v_actin_arp3)
    recipe.change_topology_type("Actin-Polymer#Shrinking")

def reaction_function_arp23_unbind_ATP(topology):
    """
    reaction function to dissociate an arp2/3 with ATP from a mother filament
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Remove Arp2/3 ATP")
    do_arp23_unbind(topology, recipe, True)
    return recipe

def reaction_function_arp23_unbind_ADP(topology):
    """
    reaction function to dissociate an arp2/3 with ADP from a mother filament
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Remove Arp2/3 ADP")
    do_arp23_unbind(topology, recipe, False)
    return recipe

def do_debranching(topology, recipe, with_ATP):
    """
    reaction function to detach a branch filament from arp2/3
    """
    v_arp2 = get_random_arp2(topology, with_ATP, True)
    if v_arp2 is None:
        if verbose:
            print("Couldn't find arp2 with {}".format("ATP" if with_ATP else "ADP"))
        recipe.change_topology_type("Actin-Polymer#Fail-Debranch-{}".format(
            "ATP" if with_ATP else "ADP"))
        return recipe
    actin_types = ["actin#branch_1", "actin#branch_ATP_1", 
                   "actin#branch_barbed_1", "actin#branch_barbed_ATP_1"]
    v_actin1 = ReaddyUtil.get_neighbor_of_types(
        topology, v_arp2, actin_types, [])
    if v_actin1 is None:
        raise Exception(f"Failed to find first branch actin\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    recipe.remove_edge(v_arp2, v_actin1)
    ReaddyUtil.set_flags(topology, recipe, v_arp2, [], ["branched"], True)
    pt_actin1 = topology.particle_type_of_vertex(v_actin1)
    if "barbed" in pt_actin1: 
        # branch is a monomer
        recipe.change_particle_type(v_actin1, "actin#free{}".format(
            "_ATP" if "ATP" in pt_actin1 else ""))
    else: 
        # branch is a filament
        ReaddyUtil.set_flags(topology, recipe, v_actin1,
            ["pointed"], ["branch"], True)
    recipe.change_topology_type("Actin-Polymer#Shrinking")

def reaction_function_debranching_ATP(topology):
    """
    reaction function to detach a branch filament from arp2/3 with ATP
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Debranching ATP")
    do_debranching(topology, recipe, True)
    return recipe

def reaction_function_debranching_ADP(topology):
    """
    reaction function to detach a branch filament from arp2/3 with ADP
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Debranching ADP")
    do_debranching(topology, recipe, False)
    return recipe

def reaction_function_finish_cap_bind(topology):
    """
    reaction function for adding a capping protein
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Finish Cap Bind")
    v_new = ReaddyUtil.get_first_vertex_of_types(topology, ["cap#new"])
    if v_new is None:
        raise Exception(f"Failed to find new cap\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    ReaddyUtil.set_flags(topology, recipe, v_new, ["bound"], ["new"], True)
    recipe.change_topology_type("Actin-Polymer")
    return recipe

def reaction_function_cap_unbind(topology):
    """
    reaction function to detach capping protein from a barbed end
    """
    recipe = readdy.StructuralReactionRecipe(topology)
    if verbose:
        print("Remove Cap")
    v_cap = ReaddyUtil.get_random_vertex_of_types(topology, ["cap#bound"])
    if v_cap is None:
        if verbose:
            print("Couldn't find cap")
        recipe.change_topology_type("Actin-Polymer#Fail-Cap-Unbind")
        return recipe
    v_actin = ReaddyUtil.get_neighbor_of_types(
        topology, v_cap, 
        get_all_polymer_actin_types("actin")
        + get_all_polymer_actin_types("actin#ATP")
        + ["actin#branch_1", "actin#branch_ATP_1"], [])
    if v_actin is None:
        raise Exception(f"Failed to find actin bound to cap\n"
            f"{ReaddyUtil.topology_to_string(topology)}")
    recipe.remove_edge(v_cap, v_actin)
    ReaddyUtil.set_flags(topology, recipe, v_cap, [], ["bound"], True)
    ReaddyUtil.set_flags(topology, recipe, v_actin, ["barbed"], [], True)
    recipe.change_topology_type("Actin-Polymer#Shrinking")
    return recipe

def rate_function_reverse_dimerize(topology):
    """
    rate function for a dimer falling apart
    """
    return rates["dimerize_reverse"]

def rate_function_reverse_trimerize(topology):
    """
    rate function for removing ATP-actin from a trimer
    """
    return rates["trimerize_reverse"]

def rate_function_barbed_shrink_ATP(topology):
    """
    rate function for removing ATP-actin from the barbed end
    """
    return rates["barbed_shrink_ATP"]

def rate_function_barbed_shrink_ADP(topology):
    """
    rate function for removing ADP-actin from the barbed end
    """
    return rates["barbed_shrink_ADP"]

def rate_function_pointed_shrink_ATP(topology):
    """
    rate function for removing ATP-actin from the pointed end
    """
    return rates["pointed_shrink_ATP"]

def rate_function_pointed_shrink_ADP(topology):
    """
    rate function for removing ADP-actin from the pointed end
    """
    return rates["pointed_shrink_ADP"]

def rate_function_hydrolyze_actin(topology):
    """
    rate function for hydrolyzing filamentous ATP-actin to ADP-actin
    """
    return rates["hydrolysis_actin"]

def rate_function_hydrolyze_arp(topology):
    """
    rate function for hydrolyzing bound arp2/3
    """
    return rates["hydrolysis_arp"]

def rate_function_nucleotide_exchange_actin(topology):
    """
    rate function for exchanging an ATP for ADP in free actin
    """
    return rates["nucleotide_exchange_actin"]

def rate_function_nucleotide_exchange_arp(topology):
    """
    rate function for exchanging an ATP for ADP in free Arp2/3
    """
    return rates["nucleotide_exchange_arp"]

def rate_function_debranching_ATP(topology):
    """
    rate function for dissociation of a daughter filament from an arp2/3 with ATP
    """
    return rates["debranching_ATP"]

def rate_function_debranching_ADP(topology):
    """
    rate function for dissociation of a daughter filament from an arp2/3 with ADP
    """
    return rates["debranching_ADP"]

def rate_function_arp23_unbind_ATP(topology):
    """
    rate function for dissociation of bound arp2/3 with ATP
    """
    return rates["arp_unbind_ATP"]

def rate_function_arp23_unbind_ADP(topology):
    """
    rate function for dissociation of bound arp2/3 with ADP
    """
    return rates["arp_unbind_ADP"]

def rate_function_cap_unbind(topology):
    """
    rate function for dissociation of capping protein
    """
    return rates["cap_unbind"]

def add_bonds_between_actins(force_constant, system, util):
    """
    add bonds between actins
    """
    bond_length = ActinStructure.actin_to_actin_distance()
    util.add_polymer_bond_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], 0,
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 1,
        force_constant, bond_length, system
    )
    util.add_bond(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2", "actin#barbed_2", "actin#barbed_ATP_2"],
        force_constant, bond_length, system
    )
    util.add_polymer_bond_1D( # temporary during growth reactions
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_",
         "actin#barbed_", "actin#barbed_ATP_"], 0,
        ["actin#new", "actin#new_ATP"], None,
        force_constant, bond_length, system
    )
    util.add_bond( # temporary during growth reactions
        ["actin#branch_1", "actin#branch_ATP_1",
         "actin#branch_barbed_1", "actin#branch_barbed_ATP_1",],
        ["actin#new", "actin#new_ATP"],
        force_constant, bond_length, system
    )

def add_filament_twist_angles(force_constant, system, util):
    """
    add angles for filament twist and cohesiveness
    """
    angle = ActinStructure.actin_to_actin_angle()
    util.add_polymer_angle_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], -1,
        ["actin#", "actin#ATP_"], 0,
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 1,
        force_constant, angle, system
    )
    util.add_angle(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["actin#3", "actin#ATP_3", "actin#barbed_3", "actin#barbed_ATP_3"],
        force_constant, angle, system
    )

def add_filament_twist_dihedrals(force_constant, system, util):
    """
    add dihedrals for filament twist and cohesiveness
    """
    angle = ActinStructure.actin_to_actin_dihedral_angle()
    util.add_polymer_dihedral_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], -1,
        ["actin#", "actin#ATP_"], 0,
        ["actin#", "actin#ATP_"], 1,
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 2,
        force_constant, angle, system
    )
    util.add_dihedral(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["actin#3", "actin#ATP_3"],
        ["actin#1", "actin#ATP_1", "actin#barbed_1", "actin#barbed_ATP_1"],
        force_constant, angle, system
    )

def add_branch_bonds(force_constant, system, util):
    """
    add bonds between arp2, arp3, and actins
    """
    util.add_polymer_bond_1D( # mother filament actin to arp2 bonds
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], 0,
        ["arp2", "arp2#branched"], None,
        force_constant, ActinStructure.arp2_to_mother_distance(), system
    )
    util.add_polymer_bond_1D( # mother filament actin to arp3 bonds
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 0,
        ["arp3", "arp3#ATP", "arp3#new", "arp3#new_ATP"], None,
        force_constant, ActinStructure.arp3_to_mother_distance(), system
    )
    util.add_bond( # arp2 to arp3 bonds
        ["arp2", "arp2#branched"],
        ["arp3", "arp3#ATP", "arp3#new", "arp3#new_ATP"],
        force_constant, ActinStructure.arp2_to_arp3_distance(), system
    )
    util.add_bond( # arp2 to daughter filament actin bonds
        ["arp2#branched"],
        ["actin#branch_1", "actin#branch_ATP_1",
         "actin#branch_barbed_1", "actin#branch_barbed_ATP_1",
         "actin#new", "actin#new_ATP"],
        force_constant, ActinStructure.arp2_to_daughter_distance(), system
    )

def add_branch_angles(force_constant, system, util):
    """
    add angles for branching
    """
    util.add_angle(
        ["arp3", "arp3#ATP"],
        ["arp2#branched"],
        ["actin#branch_1", "actin#branch_ATP_1",
         "actin#branch_barbed_1", "actin#branch_barbed_ATP_1"],
        force_constant, ActinStructure.arp3_arp2_daughter1_angle(), system
    )
    util.add_polymer_angle_1D(
        ["arp2", "arp2#branched"], None,
        ["actin#", "actin#ATP_"], 0,
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 1,
        force_constant, ActinStructure.arp2_daughter1_daughter2_angle(), system
    )
    angle = ActinStructure.mother1_mother2_arp3_angle()
    util.add_polymer_angle_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], 0,
        ["actin#", "actin#ATP_"], 1,
        ["arp3", "arp3#ATP"], None,
        force_constant, angle, system
    )
    util.add_angle(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["arp3", "arp3#ATP"],
        force_constant, angle, system
    )
    util.add_polymer_angle_1D(
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 1,
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], 0,
        ["arp3", "arp3#ATP"], None,
        force_constant, ActinStructure.mother3_mother2_arp3_angle(), system
    )
    angle = ActinStructure.mother0_mother1_arp2_angle()
    util.add_polymer_angle_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], 0,
        ["actin#", "actin#ATP_"], 1,
        ["arp2", "arp2#branched"], None,
        force_constant, angle, system
    )
    util.add_angle(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["arp2", "arp2#branched"],
        force_constant, angle, system
    )

def add_branch_dihedrals(force_constant, system, util):
    """
    add dihedrals for branching
    """
    # mother to arp
    angle = ActinStructure.mother4_mother3_mother2_arp3_dihedral_angle()
    util.add_polymer_dihedral_1D(
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 1,
        ["actin#", "actin#ATP_"], 0,
        ["actin#", "actin#ATP_"], -1,
        ["arp3", "arp3#ATP"], None,
        force_constant, angle, system
    )
    angle = ActinStructure.mother_mother0_mother1_arp2_dihedral_angle()
    util.add_polymer_dihedral_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], -1,
        ["actin#", "actin#ATP_"], 0,
        ["actin#", "actin#ATP_"], 1,
        ["arp2", "arp2#branched"], None,
        force_constant, angle, system
    )
    util.add_dihedral(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["actin#3", "actin#ATP_3"],
        ["arp2", "arp2#branched"],
        force_constant, angle, system
    )
    util.add_polymer_dihedral_1D(
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 1,
        ["actin#", "actin#ATP_"], 0,
        ["arp3", "arp3#ATP"], None,
        ["arp2#branched", "arp2"], None,
        force_constant, ActinStructure.mother3_mother2_arp3_arp2_dihedral_angle(), system
    )
    # arp ring
    angle = ActinStructure.mother1_mother2_arp3_arp2_dihedral_angle()
    util.add_polymer_dihedral_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], 0,
        ["actin#", "actin#ATP_"], 1,
        ["arp3", "arp3#ATP"], None,
        ["arp2", "arp2#branched"], None,
        force_constant, angle, system
    )
    util.add_dihedral(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["arp3", "arp3#ATP"],
        ["arp2", "arp2#branched"],
        force_constant, angle, system
    )
    angle = ActinStructure.arp2_mother1_mother2_arp3_dihedral_angle()
    util.add_polymer_dihedral_1D(
        ["arp2", "arp2#branched"], None,
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], 0,
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 1,
        ["arp3", "arp3#ATP"], None,
        force_constant, angle, system
    )
    util.add_dihedral(
        ["arp2", "arp2#branched"],
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2", "actin#barbed_2", "actin#barbed_ATP_2"],
        ["arp3", "arp3#ATP"],
        force_constant, angle, system
    )
    # arp to daughter
    util.add_dihedral(
        ["arp3", "arp3#ATP"],
        ["arp2#branched"],
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2", "actin#barbed_2", "actin#barbed_ATP_2"],
        force_constant, 
        ActinStructure.arp3_arp2_daughter1_daughter2_dihedral_angle(), 
        system,
    )
    util.add_dihedral(
        ["arp2#branched"],
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["actin#3", "actin#ATP_3", "actin#barbed_3", "actin#barbed_ATP_3"],
        force_constant, 
        ActinStructure.arp2_daughter1_daughter2_daughter3_dihedral_angle(), 
        system,
    )
    # mother to daughter
    angle = ActinStructure.mother0_mother1_arp2_daughter1_dihedral_angle()
    util.add_polymer_dihedral_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], -1,
        ["actin#", "actin#ATP_"], 0,
        ["arp2#branched"], None,
        ["actin#branch_1", "actin#branch_ATP_1",
         "actin#branch_barbed_1", "actin#branch_barbed_ATP_1"], None,
        force_constant, angle, system
    )
    util.add_dihedral(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["arp2#branched"],
        ["actin#branch_1", "actin#branch_ATP_1",
         "actin#branch_barbed_1", "actin#branch_barbed_ATP_1"],
        force_constant, angle, system
    )
    util.add_polymer_dihedral_1D(
        ["actin#", "actin#ATP_", "actin#barbed_", "actin#barbed_ATP_"], 0,
        ["arp3", "arp3#ATP"], None,
        ["arp2#branched"], None,
        ["actin#branch_1", "actin#branch_ATP_1",
         "actin#branch_barbed_1", "actin#branch_barbed_ATP_1"], None,
        force_constant, ActinStructure.mother2_arp3_arp2_daughter1_dihedral_angle(), system
    )
    util.add_dihedral(
        ["actin#1", "actin#ATP_1", "actin#pointed_1", "actin#pointed_ATP_1",
         "actin#branch_1", "actin#branch_ATP_1",
         "actin#2", "actin#ATP_2", "actin#pointed_2", "actin#pointed_ATP_2",
         "actin#3", "actin#ATP_3", "actin#pointed_3", "actin#pointed_ATP_3",],
        ["arp2#branched"],
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2",
         "actin#barbed_2", "actin#barbed_ATP_2"],
        force_constant, 
        ActinStructure.mother1_arp2_daughter1_daughter2_dihedral_angle(), 
        system,
    )

def add_cap_bonds(force_constant, system, util):
    """
    add capping protein to actin bonds
    """
    util.add_polymer_bond_1D(
        ["actin#", "actin#ATP_"], 0,
        ["cap#bound", "cap#new"], None,
        force_constant, ActinStructure.actin_to_actin_distance() + 1.0, system
    )

def add_cap_angles(force_constant, system, util):
    """
    add angles for capping protein
    """
    angle = ActinStructure.actin_to_actin_angle()
    util.add_polymer_angle_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], 0,
        ["actin#", "actin#ATP_"], 1,
        ["cap#bound"], None,
        force_constant, angle, system
    )
    util.add_angle(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["cap#bound"],
        force_constant, angle, system
    )

def add_cap_dihedrals(force_constant, system, util):
    """
    add dihedrals for capping protein
    """
    angle = ActinStructure.actin_to_actin_dihedral_angle()
    util.add_polymer_dihedral_1D(
        ["actin#", "actin#ATP_", "actin#pointed_", "actin#pointed_ATP_"], -1,
        ["actin#", "actin#ATP_"], 0,
        ["actin#", "actin#ATP_"], 1,
        ["cap#bound"], None,
        force_constant, angle, system
    )
    util.add_dihedral(
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["actin#2", "actin#ATP_2"],
        ["actin#3", "actin#ATP_3"],
        ["cap#bound"],
        force_constant, angle, system
    )
    util.add_dihedral(
        ["arp3", "arp3#ATP"],
        ["arp2#branched"],
        ["actin#branch_1", "actin#branch_ATP_1"],
        ["cap#bound"],
        force_constant, 
        ActinStructure.arp3_arp2_daughter1_daughter2_dihedral_angle(), 
        system,
    )

def add_repulsions(force_constant, system, util):
    """
    add repulsions
    """
    util.add_repulsion(
        ["actin#pointed_1", "actin#pointed_ATP_1", "actin#pointed_2",
         "actin#pointed_ATP_2", "actin#pointed_3", "actin#pointed_ATP_3",
         "actin#1", "actin#ATP_1", "actin#2", "actin#ATP_2",
         "actin#3", "actin#ATP_3",
         "actin#branch_1", "actin#branch_ATP_1",
         "actin#branch_barbed_1", "actin#branch_barbed_ATP_1",
         "actin#barbed_1", "actin#barbed_ATP_1", "actin#barbed_2",
         "actin#barbed_ATP_2", "actin#barbed_3", "actin#barbed_ATP_3",
         "arp2", "arp2#branched", "arp3", "arp3#ATP", "cap", "cap#bound",
         "actin#free", "actin#free_ATP"],
        ["actin#pointed_1", "actin#pointed_ATP_1", "actin#pointed_2",
         "actin#pointed_ATP_2", "actin#pointed_3", "actin#pointed_ATP_3",
         "actin#1", "actin#ATP_1", "actin#2", "actin#ATP_2",
         "actin#3", "actin#ATP_3",
         "actin#branch_1", "actin#branch_ATP_1",
         "actin#branch_barbed_1", "actin#branch_barbed_ATP_1",
         "actin#barbed_1", "actin#barbed_ATP_1", "actin#barbed_2",
         "actin#barbed_ATP_2", "actin#barbed_3", "actin#barbed_ATP_3",
         "arp2", "arp2#branched", "arp3", "arp3#ATP", "cap", "cap#bound",
         "actin#free", "actin#free_ATP"],
        force_constant, ActinStructure.actin_to_actin_repulsion_distance(), system
    )

def add_dimerize_reaction(system, rate, reaction_distance):
    """
    attach two monomers
    """
    system.topologies.add_spatial_reaction(
        "Dimerize: Actin-Monomer(actin#free_ATP) + {}".format(
        "Actin-Monomer(actin#free_ATP) -> {}".format(
        "Actin-Dimer(actin#pointed_ATP_1--actin#barbed_ATP_2)")),
        rate=rate, radius=reaction_distance
    )

def add_dimerize_reverse_reaction(system):
    """
    detach two monomers
    """
    system.topologies.add_structural_reaction(
        "Reverse_Dimerize",
        topology_type="Actin-Dimer",
        reaction_function=reaction_function_reverse_dimerize,
        rate_function=rate_function_reverse_dimerize
    )

def add_trimerize_reaction(system, rate, reaction_distance):
    """
    attach a monomer to a dimer
    """
    for i in range(1,4):
        system.topologies.add_spatial_reaction(
            "Trimerize{}: Actin-Dimer(actin#barbed_ATP_{}) + {}".format(i, i,
            "Actin-Monomer(actin#free_ATP) -> {}".format(
            "Actin-Trimer#Growing(actin#ATP_{}--actin#new_ATP)".format(i))),
            rate=rate, radius=reaction_distance
        )
    system.topologies.add_structural_reaction(
        "Finish_Trimerize",
        topology_type="Actin-Trimer#Growing",
        reaction_function=reaction_function_finish_trimerize,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_trimerize_reverse_reaction(system):
    """
    detach a monomer from a dimer
    """
    system.topologies.add_structural_reaction(
        "Reverse_Trimerize",
        topology_type="Actin-Trimer",
        reaction_function=reaction_function_reverse_trimerize,
        rate_function=rate_function_reverse_trimerize
    )

def add_nucleate_reaction(system, rate_ATP, rate_ADP, reaction_distance):
    """
    reversibly attach a monomer to a trimer
    """
    for i in range(1,4):
        system.topologies.add_spatial_reaction(
            "Barbed_Growth_Nucleate_ATP{}: {}".format(i,
            "Actin-Trimer(actin#barbed_ATP_{}) + {}".format(i,
            "Actin-Monomer(actin#free_ATP) -> {}".format(
            "Actin-Polymer#GrowingBarbed(actin#ATP_{}--actin#new_ATP)".format(i)))),
            rate=rate_ATP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Barbed_Growth_Nucleate_ADP{}: {}".format(i,
            "Actin-Trimer(actin#barbed_ATP_{}) + {}".format(i,
            "Actin-Monomer(actin#free) -> {}".format(
            "Actin-Polymer#GrowingBarbed(actin#ATP_{}--actin#new)".format(i)))),
            rate=rate_ADP, radius=reaction_distance
        )

def add_pointed_growth_reaction(system, rate_ATP, rate_ADP, reaction_distance):
    """
    attach a monomer to the pointed end of a filament
    """
    for i in range(1,4):
        system.topologies.add_spatial_reaction(
            "Pointed_Growth_ATP1{}: Actin-Polymer(actin#pointed_{}) + {}".format(i, i,
            "Actin-Monomer(actin#free_ATP) -> {}".format(
            "Actin-Polymer#GrowingPointed(actin#{}--actin#new_ATP)".format(i))),
            rate=rate_ATP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Pointed_Growth_ATP2{}: Actin-Polymer(actin#pointed_ATP_{}) + {}".format(i, i,
            "Actin-Monomer(actin#free_ATP) -> {}".format(
            "Actin-Polymer#GrowingPointed(actin#ATP_{}--actin#new_ATP)".format(i))),
            rate=rate_ATP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Pointed_Growth_ADP1{}: Actin-Polymer(actin#pointed_{}) + {}".format(i, i,
            "Actin-Monomer(actin#free) -> {}".format(
            "Actin-Polymer#GrowingPointed(actin#{}--actin#new)".format(i))),
            rate=rate_ADP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Pointed_Growth_ADP2{}: Actin-Polymer(actin#pointed_ATP_{}) + {}".format(i, i,
            "Actin-Monomer(actin#free) -> {}".format(
            "Actin-Polymer#GrowingPointed(actin#ATP_{}--actin#new)".format(i))),
            rate=rate_ADP, radius=reaction_distance
        )
    system.topologies.add_structural_reaction(
        "Finish_Pointed_Growth",
        topology_type="Actin-Polymer#GrowingPointed",
        reaction_function=reaction_function_finish_pointed_grow,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_pointed_shrink_reaction(system):
    """
    remove a monomer from the pointed end of a filament
    """
    system.topologies.add_structural_reaction(
        "Pointed_Shrink_ATP",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_pointed_shrink_ATP,
        rate_function=rate_function_pointed_shrink_ATP
    )
    system.topologies.add_structural_reaction(
        "Pointed_Shrink_ADP",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_pointed_shrink_ADP,
        rate_function=rate_function_pointed_shrink_ADP
    )
    system.topologies.add_structural_reaction(
        "Fail_Pointed_Shrink_ATP",
        topology_type="Actin-Polymer#Fail-Pointed-Shrink-ATP",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )
    system.topologies.add_structural_reaction(
        "Fail_Pointed_Shrink_ADP",
        topology_type="Actin-Polymer#Fail-Pointed-Shrink-ADP",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )
    system.topologies.add_structural_reaction(
        "Cleanup_Shrink",
        topology_type="Actin-Polymer#Shrinking",
        reaction_function=reaction_function_cleanup_shrink,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_barbed_growth_reaction(system, rate_ATP, rate_ADP, reaction_distance):
    """
    attach a monomer to the barbed end of a filament
    """
    for i in range(1,4):
        system.topologies.add_spatial_reaction(
            "Barbed_Growth_ATP1{}: Actin-Polymer(actin#barbed_{}) + {}".format(i, i,
            "Actin-Monomer(actin#free_ATP) -> {}".format(
            "Actin-Polymer#GrowingBarbed(actin#{}--actin#new_ATP)".format(i))),
            rate=rate_ATP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Barbed_Growth_ATP2{}: Actin-Polymer(actin#barbed_ATP_{}) + {}".format(i, i,
            "Actin-Monomer(actin#free_ATP) -> {}".format(
            "Actin-Polymer#GrowingBarbed(actin#ATP_{}--actin#new_ATP)".format(i))),
            rate=rate_ATP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Barbed_Growth_ADP1{}: Actin-Polymer(actin#barbed_{}) + {}".format(i, i,
            "Actin-Monomer(actin#free) -> {}".format(
            "Actin-Polymer#GrowingBarbed(actin#{}--actin#new)".format(i))),
            rate=rate_ADP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Barbed_Growth_ADP2{}: Actin-Polymer(actin#barbed_ATP_{}) + {}".format(i, i,
            "Actin-Monomer(actin#free) -> {}".format(
            "Actin-Polymer#GrowingBarbed(actin#ATP_{}--actin#new)".format(i))),
            rate=rate_ADP, radius=reaction_distance
        )
    system.topologies.add_spatial_reaction(
        "Branch_Barbed_Growth_ATP1: Actin-Polymer(actin#branch_barbed_1) + {}".format(
        "Actin-Monomer(actin#free_ATP) -> {}".format(
        "Actin-Polymer#GrowingBarbed(actin#branch_1--actin#new_ATP)")),
        rate=rate_ATP, radius=reaction_distance
    )
    system.topologies.add_spatial_reaction(
        "Branch_Barbed_Growth_ATP2: Actin-Polymer(actin#branch_barbed_ATP_1) + {}".format(
        "Actin-Monomer(actin#free_ATP) -> {}".format(
        "Actin-Polymer#GrowingBarbed(actin#branch_ATP_1--actin#new_ATP)")),
        rate=rate_ATP, radius=reaction_distance
    )
    system.topologies.add_spatial_reaction(
        "Branch_Barbed_Growth_ADP1: Actin-Polymer(actin#branch_barbed_1) + {}".format(
        "Actin-Monomer(actin#free) -> {}".format(
        "Actin-Polymer#GrowingBarbed(actin#branch_1--actin#new)")),
        rate=rate_ADP, radius=reaction_distance
    )
    system.topologies.add_spatial_reaction(
        "Branch_Barbed_Growth_ADP2: Actin-Polymer(actin#branch_barbed_ATP_1) + {}".format(
        "Actin-Monomer(actin#free) -> {}".format(
        "Actin-Polymer#GrowingBarbed(actin#branch_ATP_1--actin#new)")),
        rate=rate_ADP, radius=reaction_distance
    )
    system.topologies.add_structural_reaction(
        "Finish_Barbed_growth",
        topology_type="Actin-Polymer#GrowingBarbed",
        reaction_function=reaction_function_finish_barbed_grow,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_barbed_shrink_reaction(system):
    """
    remove a monomer from the barbed end of a filament
    """
    system.topologies.add_structural_reaction(
        "Barbed_Shrink_ATP",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_barbed_shrink_ATP,
        rate_function=rate_function_barbed_shrink_ATP
    )
    system.topologies.add_structural_reaction(
        "Barbed_Shrink_ADP",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_barbed_shrink_ADP,
        rate_function=rate_function_barbed_shrink_ADP
    )
    system.topologies.add_structural_reaction(
        "Fail_Barbed_Shrink_ATP",
        topology_type="Actin-Polymer#Fail-Barbed-Shrink-ATP",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )
    system.topologies.add_structural_reaction(
        "Fail_Barbed_Shrink_ADP",
        topology_type="Actin-Polymer#Fail-Barbed-Shrink-ADP",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_hydrolyze_reaction(system):
    """
    hydrolyze ATP
    """
    system.topologies.add_structural_reaction(
        "Hydrolysis_Actin",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_hydrolyze_actin,
        rate_function=rate_function_hydrolyze_actin
    )
    system.topologies.add_structural_reaction(
        "Fail_Hydrolysis_Actin",
        topology_type="Actin-Polymer#Fail-Hydrolysis-Actin",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )
    system.topologies.add_structural_reaction(
        "Hydrolysis_Arp",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_hydrolyze_arp,
        rate_function=rate_function_hydrolyze_arp
    )
    system.topologies.add_structural_reaction(
        "Fail_Hydrolysis_Arp",
        topology_type="Actin-Polymer#Fail-Hydrolysis-Arp",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_actin_nucleotide_exchange_reaction(system):
    """
    exchange ATP for ADP in free actin monomers
    """
    system.topologies.add_structural_reaction(
        "Nucleotide_Exchange_Actin",
        topology_type="Actin-Monomer",
        reaction_function=reaction_function_nucleotide_exchange_actin,
        rate_function=rate_function_nucleotide_exchange_actin
    )

def add_arp23_nucleotide_exchange_reaction(system):
    """
    exchange ATP for ADP in free Arp2/3 dimers
    """
    system.topologies.add_structural_reaction(
        "Nucleotide_Exchange_Arp",
        topology_type="Arp23-Dimer",
        reaction_function=reaction_function_nucleotide_exchange_arp,
        rate_function=rate_function_nucleotide_exchange_arp
    )

def add_arp23_bind_reaction(system, rate_ATP, rate_ADP, reaction_distance):
    """
    add arp2/3 along filament to start a branch
    """
    for i in range(1,4):
        system.topologies.add_spatial_reaction(
            "Arp_Bind_ATP1{}: Actin-Polymer(actin#ATP_{}) + Arp23-Dimer(arp3) -> {}".format(i, i,
            "Actin-Polymer#Branching(actin#ATP_{}--arp3#new)".format(i)),
            rate=rate_ATP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Arp_Bind_ATP2{}: Actin-Polymer(actin#ATP_{}) + Arp23-Dimer(arp3#ATP) -> {}".format(i, i,
            "Actin-Polymer#Branching(actin#ATP_{}--arp3#new_ATP)".format(i)),
            rate=rate_ATP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Arp_Bind_ADP1{}: Actin-Polymer(actin#{}) + Arp23-Dimer(arp3) -> {}".format(i, i,
            "Actin-Polymer#Branching(actin#{}--arp3#new)".format(i)),
            rate=rate_ADP, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Arp_Bind_ADP2{}: Actin-Polymer(actin#{}) + Arp23-Dimer(arp3#ATP) -> {}".format(i, i,
            "Actin-Polymer#Branching(actin#{}--arp3#new_ATP)".format(i)),
            rate=rate_ADP, radius=reaction_distance
        )
    system.topologies.add_structural_reaction(
        "Finish_Arp_Bind",
        topology_type="Actin-Polymer#Branching",
        reaction_function=reaction_function_finish_arp_bind,
        rate_function=ReaddyUtil.rate_function_infinity
    )
    system.topologies.add_structural_reaction(
        "Cleanup_Fail_Arp_Bind_ATP",
        topology_type="Actin-Polymer#Fail-Branch-ATP",
        reaction_function=reaction_function_cleanup_shrink,
        rate_function=ReaddyUtil.rate_function_infinity
    )
    system.topologies.add_structural_reaction(
        "Cleanup_Fail_Arp_Bind_ADP",
        topology_type="Actin-Polymer#Fail-Branch-ADP",
        reaction_function=reaction_function_cleanup_shrink,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_arp23_unbind_reaction(system):
    """
    remove an arp2/3 that is not nucleated
    """
    system.topologies.add_structural_reaction(
        "Arp_Unbind_ATP",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_arp23_unbind_ATP,
        rate_function=rate_function_arp23_unbind_ATP
    )
    system.topologies.add_structural_reaction(
        "Arp_Unbind_ADP",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_arp23_unbind_ADP,
        rate_function=rate_function_arp23_unbind_ADP
    )
    system.topologies.add_structural_reaction(
        "Fail_Arp_Unbind_ATP",
        topology_type="Actin-Polymer#Fail-Arp-Unbind-ATP",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )
    system.topologies.add_structural_reaction(
        "Fail_Arp_Unbind_ADP",
        topology_type="Actin-Polymer#Fail-Arp-Unbind-ADP",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_nucleate_branch_reaction(system, rate_ATP, rate_ADP, reaction_distance):
    """
    add actin to arp2/3 to begin a branch
    """
    system.topologies.add_spatial_reaction(
        "Barbed_Growth_Branch_ATP: Actin-Polymer(arp2) + Actin-Monomer(actin#free_ATP) -> {}".format(
        "Actin-Polymer#Branch-Nucleating(arp2#branched--actin#new_ATP)"),
        rate=rate_ATP, radius=reaction_distance
    )
    system.topologies.add_spatial_reaction(
        "Barbed_Growth_Branch_ADP: Actin-Polymer(arp2) + Actin-Monomer(actin#free) -> {}".format(
        "Actin-Polymer#Branch-Nucleating(arp2#branched--actin#new)"),
        rate=rate_ADP, radius=reaction_distance
    )
    system.topologies.add_structural_reaction(
        "Nucleate_Branch",
        topology_type="Actin-Polymer#Branch-Nucleating",
        reaction_function=reaction_function_finish_start_branch,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_debranch_reaction(system):
    """
    remove a branch
    """
    system.topologies.add_structural_reaction(
        "Debranch_ATP",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_debranching_ATP,
        rate_function=rate_function_debranching_ATP
    )
    system.topologies.add_structural_reaction(
        "Debranch_ADP",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_debranching_ADP,
        rate_function=rate_function_debranching_ADP
    )
    system.topologies.add_structural_reaction(
        "Fail_Debranch_ATP",
        topology_type="Actin-Polymer#Fail-Debranch-ATP",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )
    system.topologies.add_structural_reaction(
        "Fail_Debranch_ADP",
        topology_type="Actin-Polymer#Fail-Debranch-ADP",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_cap_bind_reaction(system, rate, reaction_distance):
    """
    add capping protein to a barbed end to stop growth
    """
    for i in range(1,4):
        system.topologies.add_spatial_reaction(
            "Cap_Bind1{}: Actin-Polymer(actin#barbed_{}) + Cap(cap) -> {}".format(i, i,
            "Actin-Polymer#Capping(actin#{}--cap#new)".format(i)),
            rate=rate, radius=reaction_distance
        )
        system.topologies.add_spatial_reaction(
            "Cap_Bind2{}: Actin-Polymer(actin#barbed_ATP_{}) + Cap(cap) -> {}".format(i, i,
            "Actin-Polymer#Capping(actin#ATP_{}--cap#new)".format(i)),
            rate=rate, radius=reaction_distance
        )
    system.topologies.add_structural_reaction(
        "Finish_Cap-Bind",
        topology_type="Actin-Polymer#Capping",
        reaction_function=reaction_function_finish_cap_bind,
        rate_function=ReaddyUtil.rate_function_infinity
    )

def add_cap_unbind_reaction(system):
    """
    remove capping protein
    """
    system.topologies.add_structural_reaction(
        "Cap_Unbind",
        topology_type="Actin-Polymer",
        reaction_function=reaction_function_cap_unbind,
        rate_function=rate_function_cap_unbind
    )
    system.topologies.add_structural_reaction(
        "Fail_Cap_Unbind",
        topology_type="Actin-Polymer#Fail-Cap-Unbind",
        reaction_function=ReaddyUtil.reaction_function_reset_state,
        rate_function=ReaddyUtil.rate_function_infinity
    )
