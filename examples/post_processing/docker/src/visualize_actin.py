#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from simularium_models_util.visualization import ActinVisualization


COLORS = [
    "#fee34d",
    "#f7b232",
    "#bf5736",
    "#94a7fc",
    "#ce8ec9",
    "#58606c",
    "#0ba345",
    "#9267cb",
    "#81dbe6",
    "#bd7800",
    "#bbbb99",
    "#5b79f0",
    "#89a500",
    "#da8692",
    "#418463",
    "#9f516c",
    "#00aabf",
]
    

def visualize_actin(
    dir_path: str,
    box_size: float,
    total_steps: float,
    experiment_name: str,
    periodic_boundary: bool,
    plot_bend_twist: bool,
    plot_polymerization: bool,
    save_in_one_file: bool,
    color_by_run: bool,
    visualize_edges: bool,
    visualize_normals: bool,
):
    """
    Parse an actin hdf5 (*.h5) trajectory file produced
    by the ReaDDy software and convert it into the Simularium
    visualization format with plots
    """
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    box_size = np.array(3 * [float(box_size)])
    trajectory_datas = []
    for file in os.listdir(dir_path):
        color_index = 0
        if not file.endswith(".h5"):
            continue
        file_path = os.path.join("outputs/", file)
        print(f"visualize {file_path}")
        if plot_polymerization or plot_bend_twist or visualize_edges or visualize_normals:
            (
                monomer_data,
                times,
                reactions,
            ) = ActinVisualization.shape_readdy_data_for_analysis(
                h5_file_path=file_path,
                reactions=plot_polymerization,
            )
        plots = None
        if plot_polymerization:
            print("plot polymerization")
            plots = ActinVisualization.generate_polymerization_plots(
                monomer_data, times, reactions, box_size, 10, periodic_boundary
            )
        if plot_bend_twist:
            print("plot bend twist")
            plots = ActinVisualization.generate_bend_twist_plots(
                monomer_data, times, box_size, 10, periodic_boundary
            )
        trajectory_datas.append(
            ActinVisualization.visualize_actin(
                path_to_readdy_h5=file_path,
                box_size=box_size,
                total_steps=total_steps,
                save_in_one_file=save_in_one_file,
                file_prefix=experiment_name,
                flags_to_change={
                    "pointed" : "P",
                    "barbed" : "B",
                    "mid" : "",
                    "ATP" : "",
                },
                color=COLORS[color_index] if color_by_run else "",
                visualize_edges=visualize_edges,
                visualize_normals=visualize_normals,
                monomer_data=monomer_data,
                plots=plots,
            )
        )
        if not save_in_one_file:
            ActinVisualization.save_actin(
                [trajectory_datas[len(trajectory_datas) - 1]],
                file_path
            )
        color_index += 1
        if color_index >= len(COLORS):
            color_index = 0
    if save_in_one_file:
        print("Saving in one file")
        ActinVisualization.save_actin(
            trajectory_datas,
            os.path.join("outputs/", f"{experiment_name}_combination")
        )

def main():
    parser = argparse.ArgumentParser(
        description="Parses an actin hdf5 (*.h5) trajectory file produced\
         by the ReaDDy software and converts it into the Simularium\
         visualization-data-format with plots"
    )
    parser.add_argument(
        "dir_path",
        help="the file path of the directory\
         containing the trajectories to parse",
    )
    parser.add_argument("box_size", help="width of simulation cube")
    parser.add_argument(
        "total_steps", help="total number of iterations during model run"
    )
    parser.add_argument(
        "experiment_name", 
        nargs='?',  # optional
        help="prefix of file names to exclude if saving in one file",
        default=""
    )
    parser.add_argument(
        "--periodic_boundary", 
        help="is there a periodic boundary condition?",
        dest='periodic_boundary', default=False, action='store_true'
    )
    parser.add_argument(
        "--plot_bend_twist",
        help="calculate bend/twist plots?",
        dest='plot_bend_twist', default=False, action='store_true'
    )
    parser.add_argument(
        "--plot_polymerization",
        help="calculate polymerization plots?",
        dest='plot_polymerization', default=False, action='store_true'
    )
    parser.add_argument(
        "--save_in_one_file",
        help="save all the trajectories in the directory as one simularium file?",
        dest='save_in_one_file', default=False, action='store_true'
    )
    parser.add_argument(
        "--color_by_run",
        help=(
            "if save_in_one_file, color the agents by the run? "
            "otherwise use default colors"
        ),
        dest='color_by_run', default=False, action='store_true'
    )
    parser.add_argument(
        "--vis_edges",
        help="Draw lines for the edges between particles?",
        dest='visualize_edges', default=False, action='store_true'
    )
    parser.add_argument(
        "--vis_normals",
        help="Draw lines for actin normals?",
        dest='visualize_normals', default=False, action='store_true'
    )
    args = parser.parse_args()
    visualize_actin(
        args.dir_path,
        float(args.box_size),
        float(args.total_steps),
        args.experiment_name,
        args.periodic_boundary,
        args.plot_bend_twist,
        args.plot_polymerization,
        args.save_in_one_file,
        args.color_by_run,
        args.visualize_edges,
        args.visualize_normals,
    )


if __name__ == "__main__":
    main()
