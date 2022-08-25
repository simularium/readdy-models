#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas
import argparse
import psutil

from simularium_models_util.microtubules import MicrotubulesSimulation, MICROTUBULES_REACTIONS
from simularium_models_util.visualization import MicrotubulesVisualization
from simularium_models_util import RepeatedTimer, ReaddyUtil


def report_memory_usage():
    print(f"RAM percent used: {psutil.virtual_memory()[2]}")


def main():
    parser = argparse.ArgumentParser(
        description="Runs and visualizes a ReaDDy microtubules simulation"
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
    parameters = pandas.read_excel(
        args.params_path,
        sheet_name="microtubules",
        usecols=[0, int(args.data_column)],
        dtype=object,
    )
    parameters.set_index("name", inplace=True)
    parameters.transpose()
    run_name = list(parameters)[0]
    parameters = parameters[run_name]
    # read in box size
    parameters["box_size"] = ReaddyUtil.get_box_size(parameters["box_size"])
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    parameters["name"] = "outputs/" + args.model_name + "_" + run_name
    mt_simulation = MicrotubulesSimulation(parameters, True, True)
    mt_simulation.add_random_tubulin_dimers()
    mt_simulation.add_microtubule_seed()
    rt = RepeatedTimer(600, report_memory_usage)  # every 10 min
    try:
        mt_simulation.simulation.run(
            int(parameters["total_steps"]), parameters["timestep"]
        )
        try:
            save_converter = True
            h5_path = parameters["name"] + ".h5"
            stride = 1

            viz_stepsize = max(int(parameters["total_steps"] / 1000.0), 1)
            scaled_time_step_us = parameters["timestep"] * 1e-3 * viz_stepsize

            (monomer_data, reactions, times, _,) = ReaddyUtil.monomer_data_and_reactions_from_file(
                h5_file_path=h5_path,
                stride=stride,
                timestep=0.1,
                reaction_names=MICROTUBULES_REACTIONS,
                save_pickle_file=True,
                pickle_file_path=None,
            )

            plots = MicrotubulesVisualization.generate_plots(
                monomer_data=monomer_data,
                reactions=reactions,
                times=times,
            )

            converter = MicrotubulesVisualization.visualize_microtubules(
                h5_path,
                box_size=parameters["box_size"],
                scaled_time_step_us=scaled_time_step_us,
                plots=plots,
            )
            
            converter._data = ReaddyUtil._add_edge_agents(
                traj_data=converter._data,
                monomer_data=monomer_data,
                exclude_types=["tubulinA#free",
                    "tubulinB#free"]
            )

            if save_converter:
                MicrotubulesVisualization.save(
                    converter,
                    output_path=h5_path,
                )

            converter = MicrotubulesVisualization.add_plots(
                converter,
                plots,
            )

            MicrotubulesVisualization.save(
                converter,
                output_path=h5_path,
            )
        except Exception as e:
            print("Failed viz!!!!!!!!!!\n" + str(type(e)) + " " + str(e))
            sys.exit(88888888)
    finally:
        rt.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()
