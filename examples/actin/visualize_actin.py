#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from matplotlib.pyplot import plot
import numpy as np
import copy
from typing import Dict, Any, List

from simularium_models_util.visualization import ActinVisualization, ACTIN_DISPLAY_DATA
from simularium_models_util.actin import ActinAnalyzer


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


def get_suffix_and_display_data(
    save_in_one_file: bool, 
    path_to_readdy_h5: str = "", 
    file_prefix: str = "",
    flags_to_change: Dict[str,str] = None,
    color: str = "",
):
    """
    Rename agents with trajectory suffix if saving in one file
    """
    suffix = ""
    if save_in_one_file:
        suffix = os.path.basename(path_to_readdy_h5)
        suffix = suffix[suffix.index(file_prefix) + len(file_prefix):]
        suffix = os.path.splitext(suffix)[0]
        suffix = suffix.replace("_", " ")
        suffix = suffix.strip()
        display_data = copy.deepcopy(ACTIN_DISPLAY_DATA)
        for agent_type in display_data:
            base_type = display_data[agent_type].name
            state = ""
            if "#" in base_type:
                state = base_type[base_type.index('#') + 1:]
                if flags_to_change is not None:
                    for flag in flags_to_change:
                        state = state.replace(flag, flags_to_change[flag])
                state = state.replace("_", " ")
                state = state.strip()
                if len(state) > 0:
                    state = ":" + state
                base_type = base_type[:base_type.index('#')]
            new_display_name = suffix + "#" + base_type + state
            display_data[agent_type].name = new_display_name
            if color:
                display_data[agent_type].color = color
    else:
        display_data = ACTIN_DISPLAY_DATA
    return suffix, display_data


def combine_plots(
    plots: Dict[str,Dict[str,Any]], 
    exclude_trace_keywords: List[str] = None
) -> Dict[str,Any]:
    """
    Combine the plots from multiple runs
    """
    if exclude_trace_keywords is None:
        exclude_trace_keywords = []
    result = {}
    for run_name in plots:
        for plot_type in plots[run_name]:
            if plot_type not in result:
                result[plot_type] = {}
            for plot_index in range(len(plots[run_name][plot_type])):
                plot = plots[run_name][plot_type][plot_index]
                if plot.title not in result[plot_type]:
                    result[plot_type][plot.title] = copy.deepcopy(plot)
                    result[plot_type][plot.title].ytraces = {}
                for y_trace_name in plot.ytraces:
                    skip = False
                    for keyword in exclude_trace_keywords:
                        if keyword in y_trace_name:
                            skip = True
                            break
                    if skip:
                        continue
                    new_trace_name = f"{run_name} : {y_trace_name}"
                    result[plot_type][plot.title].ytraces[new_trace_name] = copy.copy(plot.ytraces[y_trace_name])
    for plot_type in result:
        result[plot_type] = [plot for _, plot in result[plot_type].items()]
    return result
    

def visualize_actin(
    actin_number_types: int,
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
    color_index = 0
    plots = {}
    for file in os.listdir(dir_path):
        if not file.endswith(".h5"):
            continue
        file_path = os.path.join(dir_path, file)
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
            if plot_bend_twist or visualize_normals:
                normals, axis_positions = ActinAnalyzer.get_normals_and_axis_positions(
                    monomer_data, box_size, periodic_boundary
                )
        traj_plots = None
        if plot_polymerization:
            print("plot polymerization")
            traj_plots = ActinVisualization.generate_polymerization_plots(
                int(args.actin_number_types), monomer_data, times, reactions, box_size, periodic_boundary, traj_plots
            )
        if plot_bend_twist:
            print("plot bend twist")
            traj_plots = ActinVisualization.generate_bend_twist_plots(
                monomer_data, times, box_size, normals, axis_positions, periodic_boundary, traj_plots
            )
        color = COLORS[color_index] if color_by_run else ""
        suffix, display_data = get_suffix_and_display_data(
            save_in_one_file=save_in_one_file,
            path_to_readdy_h5=file_path,
            file_prefix=experiment_name,
            flags_to_change={
                "pointed" : "P",
                "barbed" : "B",
                "mid" : "",
                "ATP" : "",
            },
            color=color,
        )
        trajectory_datas.append(
            ActinVisualization.visualize_actin(
                actin_number_types=actin_number_types,
                path_to_readdy_h5=file_path,
                box_size=box_size,
                total_steps=total_steps,
                suffix=suffix,
                display_data=display_data,
                color=color,
                visualize_edges=visualize_edges,
                visualize_normals=visualize_normals,
                monomer_data=monomer_data,
                normals=normals, 
                axis_positions=axis_positions,
                plots=traj_plots if not save_in_one_file else None,
            )
        )
        if save_in_one_file:
            plots[suffix] = traj_plots
        else:
            ActinVisualization.save_actin(
                [trajectory_datas[len(trajectory_datas) - 1]],
                os.path.join("outputs/", file),
                traj_plots,
            )
        color_index += 1
        if color_index >= len(COLORS):
            color_index = 0
    if save_in_one_file:
        print("Saving in one file")
        exclude_y_trace_keywords = ["Start", "Mid", "+ std", "- std"] if plot_bend_twist else []
        combo_plots = combine_plots(plots, exclude_y_trace_keywords)
        ActinVisualization.save_actin(
            trajectory_datas,
            os.path.join("outputs/", f"{experiment_name}_combo"),
            combo_plots,
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
    parser.add_argument(
        "actin_number_types",
        help="number of possible actin monomer types. can be either 3 or 5.",
    )
    args = parser.parse_args()
    visualize_actin(
        args.actin_number_types,
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
