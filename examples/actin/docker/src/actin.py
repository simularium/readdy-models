#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import pandas
import psutil

from subcell_analysis.readdy import (
    ReaddyLoader, 
    ReaddyPostProcessor,
)
from simulariumio import BinaryWriter

from simularium_readdy_models.actin import (
    FiberData,
    ActinSimulation,
    ActinGenerator,
    ActinTestData,
    ActinStructure,
)
from simularium_readdy_models.visualization import ActinVisualization
from simularium_readdy_models import ReaddyUtil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy branched actin simulation"
    )
    parser.add_argument(
        "params_path", help="the file path of an excel file with parameters"
    )
    parser.add_argument(
        "data_column", help="the column index for the parameter set to use"
    )
    parser.add_argument(
        "model_name", help="prefix for output file names", nargs="?", default=""
    )
    parser.add_argument(
        "replicate", help="which replicate?", nargs="?", default=""
    )
    parser.add_argument('--save_pickle', action=argparse.BooleanOptionalAction)
    parser.set_defaults(save_pickle=False)
    return parser.parse_args()


def setup_parameters(args):
    parameters = pandas.read_excel(
        args.params_path,
        sheet_name="actin",
        usecols=[0, int(args.data_column)],
        dtype=object,
    )
    parameters.set_index("name", inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    parameters["box_size"] = ReaddyUtil.get_box_size(parameters["box_size"])
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    parameters["name"] = (
        "outputs/" + 
        args.model_name + "_" + 
        str(run_name) + 
        ("_" + args.replicate if args.replicate else "")
    )
    return parameters


def config_init_conditions(actin_simulation):
    actin_simulation.add_obstacles()
    actin_simulation.add_random_monomers()
    actin_simulation.add_random_linear_fibers(use_uuids=False)
    longitudinal_bonds = bool(actin_simulation.parameters.get("longitudinal_bonds", True))
    if bool(actin_simulation.parameters.get("orthogonal_seed", False)):
        print("Starting with orthogonal seed")
        monomers = ActinGenerator.get_monomers(
            fibers_data=[
                FiberData(
                    28,
                    [
                        np.array([-250, 0, 0]),
                        np.array([250, 0, 0]),
                    ],
                    "Actin-Polymer",
                )
            ], 
            use_uuids=False, 
            start_normal=np.array([0., 1., 0.]), 
            longitudinal_bonds=longitudinal_bonds,
        )
        monomers = ActinGenerator.setup_fixed_monomers(monomers, actin_simulation.parameters)
        actin_simulation.add_monomers_from_data(monomers)
    if bool(actin_simulation.parameters.get("branched_seed", False)):
        print("Starting with branched seed")
        actin_simulation.add_monomers_from_data(
            ActinGenerator.get_monomers(
                fibers_data=ActinTestData.simple_branched_actin_fiber(),
                use_uuids=False,
                longitudinal_bonds=longitudinal_bonds,
            )
        )


def report_hardware_usage():
    avg_load = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
    print(
        f"AVG load: {avg_load[0]} last min, {avg_load[1]} last 5 min, {avg_load[2]} last 15 min\n"
        f"RAM % used: {psutil.virtual_memory()[2]}\n"
        f"CPU % used: {psutil.cpu_percent()}\n"
        f"Disk % used: {psutil.disk_usage('/').percent}\n"
    )


def analyze_results(parameters, save_pickle=False):
    # get analysis parameters
    plot_actin_compression = parameters.get("plot_actin_compression", False) 
    visualize_edges = parameters.get("visualize_edges", False) 
    visualize_normals = parameters.get("visualize_normals", False) 
    visualize_control_pts = parameters.get("visualize_control_pts", False)
    
    # convert to simularium
    traj_data = ActinVisualization.simularium_trajectory(
        path_to_readdy_h5=parameters["name"] + ".h5",
        box_size=parameters["box_size"],
        total_steps=parameters["total_steps"],
        time_multiplier=1e-3,  # assume 1e3 recorded steps
        longitudinal_bonds=bool(parameters.get("longitudinal_bonds", True)),
    )
    
    # load different views of ReaDDy data
    post_processor = None
    fiber_chain_ids = None
    axis_positions = None
    new_chain_ids = None
    if visualize_normals or visualize_control_pts or plot_actin_compression or visualize_edges: 
        periodic_boundary = parameters.get("periodic_boundary", False) 
        post_processor = ReaddyPostProcessor(
            trajectory=ReaddyLoader(
                h5_file_path=parameters["name"] + ".h5",
                min_time_ix=0,
                max_time_ix=-1,
                time_inc=1,
                timestep=100.0,
                save_pickle_file=save_pickle,
            ).trajectory(),
            box_size=parameters["box_size"],
            periodic_boundary=periodic_boundary,
        )
        if visualize_normals or visualize_control_pts or plot_actin_compression:
            fiber_chain_ids = post_processor.linear_fiber_chain_ids(
                start_particle_phrases=["pointed"],
                other_particle_types=[
                    "actin#",
                    "actin#ATP_",
                    "actin#mid_",
                    "actin#mid_ATP_",
                    "actin#fixed_",
                    "actin#fixed_ATP_",
                    "actin#mid_fixed_",
                    "actin#mid_fixed_ATP_",
                    "actin#barbed_",
                    "actin#barbed_ATP_",
                    "actin#fixed_barbed_",
                    "actin#fixed_barbed_ATP_",
                ],
                polymer_number_range=5,
            )
            if visualize_normals or visualize_control_pts:
                axis_positions, new_chain_ids = post_processor.linear_fiber_axis_positions(
                    fiber_chain_ids=fiber_chain_ids,
                    ideal_positions=ActinStructure.mother_positions[2:5],
                    ideal_vector_to_axis=ActinStructure.vector_to_axis(),
                )
    
    # create plots
    if plot_actin_compression:
        print("plot actin compression")
        traj_data.plots = ActinVisualization.generate_actin_compression_plots(
            post_processor,
            fiber_chain_ids,
            temperature_c=parameters["temperature_C"],
        )

    # add annotation objects to the spatial data
    traj_data = ActinVisualization.add_spatial_annotations(
        traj_data,
        post_processor,
        visualize_edges,
        visualize_normals,
        visualize_control_pts,
        new_chain_ids,
        axis_positions,
    )
    
    # save simularium file
    BinaryWriter.save(
        trajectory_data=traj_data,
        output_path=parameters["name"] + ".h5",
        validate_ids=False,
    )


def main():
    args = parse_args()
    parameters = setup_parameters(args)
    actin_simulation = ActinSimulation(
        parameters=parameters, 
        record=True, 
        save_checkpoints=False,
    )
    config_init_conditions(actin_simulation)
    actin_simulation.simulation.run(
        n_steps=int(actin_simulation.parameters["total_steps"]), 
        timestep=actin_simulation.parameters.get("internal_timestep", 0.1),
        show_summary=False,
    )
    report_hardware_usage()
    analyze_results(actin_simulation.parameters, args.save_pickle)


if __name__ == "__main__":
    main()
