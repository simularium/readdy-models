#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import pandas
import psutil

from simularium_models_util.actin import (
    FiberData,
    ActinSimulation,
    ActinGenerator,
    ActinTestData,
    ActinAnalyzer,
)
from simularium_models_util.visualization import ActinVisualization, ACTIN_DISPLAY_DATA
from simularium_models_util import ReaddyUtil


def report_hardware_usage():
    avg_load = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
    print(
        f"AVG load: {avg_load[0]} last min, {avg_load[1]} last 5 min, {avg_load[2]} last 15 min\n"
        f"RAM % used: {psutil.virtual_memory()[2]}\n"
        f"CPU % used: {psutil.cpu_percent()}\n"
        f"Disk % used: {psutil.disk_usage('/').percent}\n"
    )


def main():
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
    args = parser.parse_args()
    print("finished args")
    parameters = pandas.read_excel(
        args.params_path,
        sheet_name="actin",
        usecols=[0, int(args.data_column)],
        dtype=object,
    )
    print(f"parameter excel: {parameters}")
    parameters.set_index("name", inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    # read in box size
    parameters["box_size"] = ReaddyUtil.get_box_size(parameters["box_size"])
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    parameters["name"] = "outputs/" + args.model_name + "_" + str(run_name)
    # parameters["box_size"] = float(parameters["box_size"])
    actin_simulation = ActinSimulation(parameters, True, False)
    actin_simulation.add_obstacles()
    actin_simulation.add_random_monomers()
    actin_simulation.add_random_linear_fibers(use_uuids=False)
    longitudinal_bonds = bool(parameters["longitudinal_bonds"])
    if parameters["orthogonal_seed"]:
        print("Starting with orthogonal seed")
        fiber_data = [
            FiberData(
                28,
                [
                    np.array([-250, 0, 0]),
                    np.array([250, 0, 0]),
                ],
                "Actin-Polymer",
            )
        ]
        monomers = ActinGenerator.get_monomers(
            fiber_data, 
            use_uuids=False, 
            start_normal=np.array([0., 1., 0.]), 
            longitudinal_bonds=longitudinal_bonds,
        )
        monomers = ActinGenerator.setup_fixed_monomers(monomers, parameters)
        actin_simulation.add_monomers_from_data(monomers)
    print("success orthogonal if loop")
    if parameters["branched_seed"]:
        print("Starting with branched seed")
        actin_simulation.add_monomers_from_data(
            ActinGenerator.get_monomers(
                ActinTestData.simple_branched_actin_fiber(),
                use_uuids=False,
                longitudinal_bonds=longitudinal_bonds,
            )
        )
    actin_simulation.simulation.run(
        int(parameters["total_steps"]), parameters["internal_timestep"]
    )
    report_hardware_usage()
    monomer_data = None
    times = None
    reactions = None
    normals = None 
    axis_positions = None
    if (
        parameters["plot_polymerization"] 
        or parameters["plot_bend_twist"] 
        or parameters["visualize_edges"] 
        or parameters["visualize_normals"]):
        (
            monomer_data,
            times,
            reactions,
        ) = ActinVisualization.shape_readdy_data_for_analysis(
            h5_file_path=parameters["name"] + ".h5",
            reactions=parameters["plot_polymerization"],
        )
        if parameters["plot_bend_twist"] or parameters["visualize_normals"]:
            normals, axis_positions = ActinAnalyzer.analyze_normals_and_axis_positions(
                monomer_data, parameters["box_size"], parameters["periodic_boundary"]
            )
    plots = None
    if parameters["plot_polymerization"]:
        print("plot polymerization")
        plots = ActinVisualization.generate_polymerization_plots(
            monomer_data,
            times,
            reactions,
            parameters["box_size"],
            parameters["periodic_boundary"],
            plots,
        )
    if parameters["plot_bend_twist"]:
        plots = ActinVisualization.generate_bend_twist_plots(
            monomer_data,
            times,
            parameters["box_size"],
            normals,
            axis_positions,
            parameters["periodic_boundary"],
            plots,
        )
    traj_data = ActinVisualization.visualize_actin(
        path_to_readdy_h5=parameters["name"] + ".h5",
        box_size=parameters["box_size"],
        total_steps=parameters["total_steps"],
        display_data=ACTIN_DISPLAY_DATA,
        visualize_edges=parameters["visualize_edges"],
        visualize_normals=parameters["visualize_normals"],
        monomer_data=monomer_data,
        normals=normals,
        axis_positions=axis_positions,
        plots=plots,
        longitudinal_bonds=longitudinal_bonds,
    )
    ActinVisualization.save_actin(
        trajectory_datas=[traj_data],
        output_path=parameters["name"] + ".h5",
    )


if __name__ == "__main__":
    main()
