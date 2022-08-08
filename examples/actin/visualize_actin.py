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
        help="prefix of file names to exclude if saving in one file",
        default="",
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
    args = parser.parse_args()
    dir_path = args.dir_path
    box_size = np.array(3 * [float(args.box_size)])
    trajectory_datas = []
    color_index = 0
    for file in os.listdir(dir_path):
        if not file.endswith(".h5"):
            continue
        file_path = os.path.join(dir_path, file)
        print(f"visualize {file_path}")
        plots = None
        if bool(args.plot_bend_twist):
            print("plot bend twist")
            plots = ActinVisualization.generate_bend_twist_plots(
                file_path, box_size, 10, args.periodic_boundary
            )
        elif bool(args.plot_polymerization):
            print("plot polymerization")
            plots = ActinVisualization.generate_polymerization_plots(
                file_path, box_size, 10, args.periodic_boundary
            )
        trajectory_datas.append(
            ActinVisualization.visualize_actin(
                file_path,
                box_size,
                float(args.total_steps),
                args.save_in_one_file,
                args.experiment_name,
                {
                    "pointed" : "P",
                    "barbed" : "B",
                    "mid" : "",
                    "ATP" : "",
                },
                color=COLORS[color_index] if args.color_by_run else "",
                visualize_edges=args.visualize_edges,
                plots=plots,
            )
        )
        if not args.save_in_one_file:
            ActinVisualization.save_actin(
                [trajectory_datas[len(trajectory_datas) - 1]],
                file_path
            )
        color_index += 1
        if color_index >= len(COLORS):
            color_index = 0
    if args.save_in_one_file:
        print("Saving in one file")
        ActinVisualization.save_actin(
            trajectory_datas,
            os.path.join(dir_path, f"{args.experiment_name}_combination")
        )


if __name__ == "__main__":
    main()
