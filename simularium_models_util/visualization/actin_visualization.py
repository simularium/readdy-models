#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Any
import os
import copy

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import (
    TrajectoryConverter,
    MetaData,
    UnitData,
    ScatterPlotData,
    DisplayData,
    DISPLAY_TYPE,
    BinaryWriter,
    TrajectoryData,
    DimensionData,
)
from simulariumio.filters import MultiplyTimeFilter, AddAgentsFilter
from simulariumio.constants import VIZ_TYPE

from ..actin import ActinAnalyzer, ACTIN_REACTIONS
from ..common import ReaddyUtil
from tqdm import tqdm


TIMESTEP = 0.1  # ns
GROWTH_RXNS = [
    "Dimerize",
    "Trimerize",
    "Grow Pointed",
    "Grow Barbed",
    "Bind Arp2/3",
    "Start Branch",
    "Bind Cap",
]
GROUPED_GROWTH_RXNS = {
    "Dimerize Actin": ["Dimerize"],
    "Polymerize Actin": ["Trimerize", "Grow Pointed", "Grow Barbed", "Start Branch"],
    "Bind Arp2/3": ["Bind Arp2/3"],
    "Bind Cap": ["Bind Cap"],
}
STRUCTURAL_RXNS = [
    "Reverse Dimerize",
    "Reverse Trimerize",
    "Shrink Pointed",
    "Shrink Barbed",
    "Unbind Arp2/3",
    "Debranch",
    "Unbind Cap",
    "Hydrolyze Actin",
    "Hydrolyze Arp2/3",
    "Bind ATP (actin)",
    "Bind ATP (arp2/3)",
]

extra_radius = 1.5
actin_radius = 2.0 + extra_radius
arp23_radius = 2.0 + extra_radius
cap_radius = 3.0 + extra_radius
obstacle_radius = 35.0
DISPLAY_DATA = {
    "arp2": DisplayData(
        name="arp2",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=arp23_radius,
        color="#c9df8a",
    ),
    "arp2#branched": DisplayData(
        name="arp2#branched",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=arp23_radius,
        color="#c9df8a",
    ),
    "arp2#free": DisplayData(
        name="arp2#free",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=arp23_radius,
        color="#234d20",
    ),
    "arp3": DisplayData(
        name="arp3",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=arp23_radius,
        color="#36802d",
    ),
    "arp3#new": DisplayData(
        name="arp3",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=arp23_radius,
        color="#36802d",
    ),
    "arp3#ATP": DisplayData(
        name="arp3#ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=arp23_radius,
        color="#77ab59",
    ),
    "arp3#new_ATP": DisplayData(
        name="arp3#ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=arp23_radius,
        color="#77ab59",
    ),
    "cap": DisplayData(
        name="cap",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=cap_radius,
        color="#005073",
    ),
    "cap#new": DisplayData(
        name="cap",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=cap_radius,
        color="#189ad3",
    ),
    "cap#bound": DisplayData(
        name="cap#bound",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=cap_radius,
        color="#189ad3",
    ),
    "actin#free": DisplayData(
        name="actin#free",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#8d5524",
    ),
    "actin#free_ATP": DisplayData(
        name="actin#free_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#cd8500",
    ),
    "actin#new": DisplayData(
        name="actin",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#new_ATP": DisplayData(
        name="actin#ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#1": DisplayData(
        name="actin",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#2": DisplayData(
        name="actin",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#3": DisplayData(
        name="actin",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#ATP_1": DisplayData(
        name="actin#ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#ATP_2": DisplayData(
        name="actin#ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#ATP_3": DisplayData(
        name="actin#ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#mid_1": DisplayData(
        name="actin#mid",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#mid_2": DisplayData(
        name="actin#mid",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#mid_3": DisplayData(
        name="actin#mid",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#mid_ATP_1": DisplayData(
        name="actin#mid_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#mid_ATP_2": DisplayData(
        name="actin#mid_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#mid_ATP_3": DisplayData(
        name="actin#mid_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#pointed_1": DisplayData(
        name="actin#pointed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_2": DisplayData(
        name="actin#pointed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_3": DisplayData(
        name="actin#pointed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_ATP_1": DisplayData(
        name="actin#pointed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_ATP_2": DisplayData(
        name="actin#pointed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_ATP_3": DisplayData(
        name="actin#pointed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#barbed_1": DisplayData(
        name="actin#barbed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#barbed_2": DisplayData(
        name="actin#barbed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#barbed_3": DisplayData(
        name="actin#barbed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#barbed_ATP_1": DisplayData(
        name="actin#barbed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#barbed_ATP_2": DisplayData(
        name="actin#barbed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#barbed_ATP_3": DisplayData(
        name="actin#barbed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#fixed_1": DisplayData(
        name="actin#fixed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#fixed_2": DisplayData(
        name="actin#fixed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#fixed_3": DisplayData(
        name="actin#fixed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#fixed_ATP_1": DisplayData(
        name="actin#fixed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#fixed_ATP_2": DisplayData(
        name="actin#fixed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#fixed_ATP_3": DisplayData(
        name="actin#fixed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#mid_fixed_1": DisplayData(
        name="actin#mid_fixed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#mid_fixed_2": DisplayData(
        name="actin#mid_fixed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#mid_fixed_3": DisplayData(
        name="actin#mid_fixed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#bf9b30",
    ),
    "actin#mid_fixed_ATP_1": DisplayData(
        name="actin#mid_fixed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#mid_fixed_ATP_2": DisplayData(
        name="actin#mid_fixed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#mid_fixed_ATP_3": DisplayData(
        name="actin#mid_fixed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffbf00",
    ),
    "actin#pointed_fixed_1": DisplayData(
        name="actin#pointed_fixed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_fixed_2": DisplayData(
        name="actin#pointed_fixed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_fixed_3": DisplayData(
        name="actin#pointed_fixed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_fixed_ATP_1": DisplayData(
        name="actin#pointed_fixed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_fixed_ATP_2": DisplayData(
        name="actin#pointed_fixed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#pointed_fixed_ATP_3": DisplayData(
        name="actin#pointed_fixed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#fixed_barbed_1": DisplayData(
        name="actin#fixed_barbed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#fixed_barbed_2": DisplayData(
        name="actin#fixed_barbed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#fixed_barbed_3": DisplayData(
        name="actin#fixed_barbed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#fixed_barbed_ATP_1": DisplayData(
        name="actin#fixed_barbed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#fixed_barbed_ATP_2": DisplayData(
        name="actin#fixed_barbed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#fixed_barbed_ATP_3": DisplayData(
        name="actin#fixed_barbed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#branch_1": DisplayData(
        name="actin#branch",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#branch_ATP_1": DisplayData(
        name="actin#branch_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#a67c00",
    ),
    "actin#branch_barbed_1": DisplayData(
        name="actin#branch_barbed",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "actin#branch_barbed_ATP_1": DisplayData(
        name="actin#branch_barbed_ATP",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=actin_radius,
        color="#ffdc73",
    ),
    "obstacle": DisplayData(
        name="obstacle",
        display_type=DISPLAY_TYPE.SPHERE,
        radius=obstacle_radius,
        color="#666666",
    ),
}


class ActinVisualization:
    """
    visualize an actin trajectory in Simularium
    """

    @staticmethod
    def shape_readdy_data_for_analysis(
        h5_file_path, stride=1, reactions=False,
    ):
        """
        Load a file from ReaDDy 
        and shape monomer and reactions data from it
        """
        (
            monomer_data,
            reactions,
            times,
            _,
        ) = ReaddyUtil.monomer_data_and_reactions_from_file(
            h5_file_path=h5_file_path,
            stride=stride,
            timestep=0.1,
            reaction_names=ACTIN_REACTIONS if reactions else None,
            pickle_file_path=f"{h5_file_path}.dat",
            save_pickle_file=True,
        )
        return monomer_data, times, reactions

    @staticmethod
    def get_bound_monomers_plot(monomer_data, times):
        """
        Add a plot of percent actin in filaments
        """
        return ScatterPlotData(
            title="Monomers over time",
            xaxis_title="Time (µs)",
            yaxis_title="Monomers (%)",
            xtrace=times,
            ytraces={
                "Actin in filaments": 100.0
                * ActinAnalyzer.analyze_ratio_of_filamentous_to_total_actin(
                    monomer_data
                ),
                "Arp2/3 in filaments": 100.0
                * ActinAnalyzer.analyze_ratio_of_bound_to_total_arp23(monomer_data),
                "Actin in daughter filaments": 100.0
                * ActinAnalyzer.analyze_ratio_of_daughter_to_total_actin(monomer_data),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_avg_length_plot(monomer_data, times):
        """
        Add a plot of average mother and daughter filament length
        """
        return ScatterPlotData(
            title="Average length of filaments",
            xaxis_title="Time (µs)",
            yaxis_title="Average length (monomers)",
            xtrace=times,
            ytraces={
                "Mother filaments": ActinAnalyzer.analyze_average_for_series(
                    ActinAnalyzer.analyze_mother_filament_lengths(monomer_data)
                ),
                "Daughter filaments": ActinAnalyzer.analyze_average_for_series(
                    ActinAnalyzer.analyze_daughter_filament_lengths(monomer_data)
                ),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_growth_reactions_plot(reactions, times):
        """
        Add a plot of reaction events over time
        for each total growth reaction
        """
        ytraces = {}
        for total_rxn_name in GROWTH_RXNS:
            rxn_events = ReaddyUtil.analyze_reaction_count_over_time(
                reactions, total_rxn_name
            )
            if rxn_events is not None:
                ytraces[total_rxn_name] = rxn_events
        return ScatterPlotData(
            title="Growth reactions",
            xaxis_title="Time (µs)",
            yaxis_title="Reaction events",
            xtrace=times,
            ytraces=ytraces,
            render_mode="lines",
        )

    @staticmethod
    def get_structural_reactions_plot(reactions, times):
        """
        Add a plot of the number of times a structural reaction
        was triggered over time
        Note: triggered != completed, the reaction may have failed
        to find the required reactants
        """
        ytraces = {}
        for total_rxn_name in STRUCTURAL_RXNS:
            rxn_events = ReaddyUtil.analyze_reaction_count_over_time(
                reactions, total_rxn_name
            )
            if rxn_events is not None:
                ytraces[total_rxn_name] = rxn_events
        return ScatterPlotData(
            title="Structural reaction triggers",
            xaxis_title="Time (µs)",
            yaxis_title="Reactions triggered",
            xtrace=times,
            ytraces=ytraces,
            render_mode="lines",
        )

    @staticmethod
    def get_growth_reactions_vs_actin_plot(reactions, monomer_data, box_size):
        """
        Add a plot of average reaction events over time
        for each total growth reaction
        """
        ytraces = {}
        for rxn_group_name in GROUPED_GROWTH_RXNS:
            group_reaction_events = []
            for total_rxn_name in GROUPED_GROWTH_RXNS[rxn_group_name]:
                group_reaction_events.append(
                    ReaddyUtil.analyze_reaction_count_over_time(
                        reactions, total_rxn_name
                    )
                )
            if len(group_reaction_events) > 0:
                ytraces[rxn_group_name] = np.sum(
                    np.array(group_reaction_events), axis=0
                )
        return ScatterPlotData(
            title="Growth vs [actin]",
            xaxis_title="[Actin] (µM)",
            yaxis_title="Reaction events",
            xtrace=ActinAnalyzer.analyze_free_actin_concentration_over_time(
                monomer_data, box_size
            ),
            ytraces=ytraces,
            render_mode="lines",
        )

    @staticmethod
    def get_capped_ends_plot(monomer_data, times):
        """
        Add a plot of percent barbed ends that are capped
        """
        return ScatterPlotData(
            title="Capped barbed ends",
            xaxis_title="Time (µs)",
            yaxis_title="Capped ends (%)",
            xtrace=times,
            ytraces={
                "Capped ends": 100.0
                * ActinAnalyzer.analyze_ratio_of_capped_ends_to_total_ends(
                    monomer_data
                ),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_branch_angle_plot(monomer_data, box_size, periodic_boundary, times):
        """
        Add a plot of branch angle mean and std dev
        """
        angles = ActinAnalyzer.analyze_branch_angles(
            monomer_data, box_size, periodic_boundary
        )
        mean = ActinAnalyzer.analyze_average_for_series(angles)
        stddev = ActinAnalyzer.analyze_stddev_for_series(angles)
        return ScatterPlotData(
            title="Average branch angle",
            xaxis_title="Time (µs)",
            yaxis_title="Branch angle (°)",
            xtrace=times,
            ytraces={
                "Ideal": np.array(times.shape[0] * [70.9]),
                "Mean": mean,
                "Mean - std": mean - stddev,
                "Mean + std": mean + stddev,
            },
            render_mode="lines",
        )

    @staticmethod
    def get_helix_pitch_plot(monomer_data, box_size, periodic_boundary, times):
        """
        Add a plot of average helix pitch
        for both the short and long helices
        ideal Ref: http://www.jbc.org/content/266/1/1.full.pdf
        """
        return ScatterPlotData(
            title="Average helix pitch",
            xaxis_title="Time (µs)",
            yaxis_title="Pitch (nm)",
            xtrace=times,
            ytraces={
                "Ideal short pitch": np.array(times.shape[0] * [5.9]),
                "Mean short pitch": ActinAnalyzer.analyze_average_for_series(
                    ActinAnalyzer.analyze_short_helix_pitches(
                        monomer_data, box_size, periodic_boundary
                    )
                ),
                "Ideal long pitch": np.array(times.shape[0] * [72]),
                "Mean long pitch": ActinAnalyzer.analyze_average_for_series(
                    ActinAnalyzer.analyze_long_helix_pitches(
                        monomer_data, box_size, periodic_boundary
                    )
                ),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_filament_straightness_plot(
        monomer_data, box_size, periodic_boundary, times
    ):
        """
        Add a plot of how many nm each monomer is away
        from ideal position in a straight filament
        """
        return ScatterPlotData(
            title="Filament bending",
            xaxis_title="Time (µs)",
            yaxis_title="Filament bending",
            xtrace=times,
            ytraces={
                "Filament bending": (
                    ActinAnalyzer.analyze_average_for_series(
                        ActinAnalyzer.analyze_filament_straightness(
                            monomer_data, box_size, periodic_boundary
                        )
                    )
                ),
            },
            render_mode="lines",
        )

    @staticmethod
    def generate_polymerization_plots(
        monomer_data, times, reactions, box_size, periodic_boundary=True, plots=None
    ):
        """
        Use an ActinAnalyzer to generate plots of observables
        for polymerizing actin
        """
        if plots is None:
            plots = {
                "scatter": [],
                "histogram": [],
            }
        plots["scatter"] += [
            ActinVisualization.get_bound_monomers_plot(monomer_data, times),
            ActinVisualization.get_avg_length_plot(monomer_data, times),
            ActinVisualization.get_growth_reactions_plot(reactions, times),
            ActinVisualization.get_growth_reactions_vs_actin_plot(
                reactions, monomer_data, box_size
            ),
            # ActinVisualization.get_capped_ends_plot(monomer_data, times),
            ActinVisualization.get_branch_angle_plot(
                monomer_data, box_size, periodic_boundary, times
            ),
            ActinVisualization.get_helix_pitch_plot(
                monomer_data, box_size, periodic_boundary, times
            ),
            ActinVisualization.get_filament_straightness_plot(
                monomer_data, box_size, periodic_boundary, times
            ),
            ActinVisualization.get_structural_reactions_plot(reactions, times),
        ]
        return plots

    @staticmethod
    def get_total_twist_plot(monomer_data, box_size, periodic_boundary, times):
        """
        Add a plot of total twist vs end displacement
        """
        total_twist_raw, total_twist_remove_bend_raw, _ = ActinAnalyzer.analyze_total_twist(
            monomer_data, box_size, periodic_boundary
        )
        total_twist = []
        total_twist_remove_bend = []
        for t in range(len(total_twist_raw)):
            twist = 0
            for m in total_twist_raw[t]:
                twist += m
            twist_no_bend = 0
            for m in total_twist_remove_bend_raw[t]:
                twist_no_bend += m
            total_twist.append(twist)
            total_twist_remove_bend.append(twist_no_bend)
        return ScatterPlotData(
            title="Twist along filament",
            xaxis_title="Time (us)",
            yaxis_title="Twist (rotations)",
            xtrace=times,
            ytraces={
                "Total twist (degrees)": np.array(total_twist),
                "Total twist excluding bend (degrees)": np.array(total_twist_remove_bend),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_twist_per_monomer_plot(monomer_data, box_size, periodic_boundary):
        """
        Add a plot of twist vs position of the monomer in filament
        """
        total_twist, total_twist_remove_bend, filament_positions = ActinAnalyzer.analyze_total_twist(
            monomer_data, box_size, periodic_boundary
        )
        mid_time = int(len(total_twist) / 2.)
        end_time = len(total_twist) - 1
        return ScatterPlotData(
            title="Twist per monomer",
            xaxis_title="Position in filament",
            yaxis_title="Twist (rotations)",
            xtrace=filament_positions[0],
            ytraces={
                "Total twist (degrees) Start": total_twist[0],
                "Total twist (degrees) Mid": total_twist[mid_time],
                "Total twist (degrees) End": total_twist[end_time],
                "Total twist excluding bend (degrees) Start": total_twist_remove_bend[0],
                "Total twist excluding bend (degrees) Mid": total_twist_remove_bend[mid_time],
                "Total twist excluding bend (degrees) End": total_twist_remove_bend[end_time],
            },
            render_mode="lines",
        )

    @staticmethod
    def get_total_bond_length_plot(monomer_data, box_size, periodic_boundary, times):
        """
        Add a plot of bond lengths (lat and long) vs end displacement
        (normalize bond lengths relative to theoretical lengths, plot average ± std)
        """
        lateral_bond_lengths, longitudinal_bond_lengths, _ = ActinAnalyzer.analyze_bond_lengths(
            monomer_data, box_size, periodic_boundary
        )
        mean_lat = ActinAnalyzer.analyze_average_for_series(lateral_bond_lengths)
        stddev_lat = ActinAnalyzer.analyze_stddev_for_series(lateral_bond_lengths)
        mean_long = ActinAnalyzer.analyze_average_for_series(longitudinal_bond_lengths)
        stddev_long = ActinAnalyzer.analyze_stddev_for_series(longitudinal_bond_lengths)
        return ScatterPlotData(
            title="Bond lengths",
            xaxis_title="Time (us)",
            yaxis_title="Normalized bond length",
            xtrace=times,
            ytraces={
                "Lateral mean": mean_lat,
                "Lateral mean - std ": mean_lat - stddev_lat,
                "Lateral mean + std": mean_lat + stddev_lat,
                "Longitudinal mean": mean_long,
                "Longitudinal mean - std": mean_long - stddev_long,
                "Longitudinal mean + std": mean_long + stddev_long,
            },
            render_mode="lines",
        )

    @staticmethod
    def get_bond_length_per_monomer_plot(monomer_data, box_size, periodic_boundary):
        """
        Add a plot of bond lengths (lat and long) vs position of monomer in filament
        normalize bond lengths relative to theoretical lengths
        """
        lateral_bond_lengths, longitudinal_bond_lengths, filament_positions = ActinAnalyzer.analyze_bond_lengths(
            monomer_data, box_size, periodic_boundary
        )
        mid_time = int(len(lateral_bond_lengths) / 2.)
        end_time = len(lateral_bond_lengths) - 1
        return ScatterPlotData(
            title="Bond lengths",
            xaxis_title="Pointed end displacement (nm)",
            yaxis_title="Normalized bond length",
            xtrace=filament_positions[0],
            ytraces={
                "Lateral Start": lateral_bond_lengths[0],
                "Lateral Mid": lateral_bond_lengths[mid_time],
                "Lateral End": lateral_bond_lengths[end_time],
                "Longitudinal Start": longitudinal_bond_lengths[0],
                "Longitudinal Mid": longitudinal_bond_lengths[mid_time],
                "Longitudinal End": longitudinal_bond_lengths[end_time],
            },
            render_mode="lines",
        )

    @staticmethod
    def generate_bend_twist_plots(
       monomer_data, times, box_size, periodic_boundary=True, plots=None
    ):
        """
        Use an ActinAnalyzer to generate plots of observables
        for actin being bent or twisted
        """
        if plots is None:
            plots = {
                "scatter": [],
                "histogram": [],
            }
        plots["scatter"] += [
            ActinVisualization.get_total_twist_plot(
                monomer_data, box_size, periodic_boundary, times
            ),
            ActinVisualization.get_twist_per_monomer_plot(
                monomer_data, box_size, periodic_boundary
            ),
            ActinVisualization.get_total_bond_length_plot(
                monomer_data, box_size, periodic_boundary, times
            ),
            ActinVisualization.get_bond_length_per_monomer_plot(
                monomer_data, box_size, periodic_boundary
            ),
        ]
        return plots
    
    @staticmethod
    def _add_edge_agents(
        filtered_data: TrajectoryData, 
        monomer_data: List[Dict[str,Any]]
    ) -> TrajectoryData:
        """
        Add agent data for fibers to draw along the edges between particles
        """
        if monomer_data is None:
            raise Exception("Edge visualization requires monomer_data")
        # get dimensions of data
        total_steps = len(monomer_data)
        max_edges = 0
        for time_index in range(total_steps):
            n_edges = 0
            for particle_id in monomer_data[time_index]["particles"]:
                particle = monomer_data[time_index]["particles"][particle_id]
                n_edges += len(particle["neighbor_ids"])
            if n_edges > max_edges:
                max_edges = n_edges
        dimensions = DimensionData(
            total_steps=0,
            max_agents=max_edges,
            max_subpoints=2,
        )
        new_agent_data = filtered_data.agent_data.get_copy_with_increased_buffer_size(dimensions)
        # add new agents
        max_used_uid = max(list(np.unique(filtered_data.agent_data.unique_ids)))
        print("Processing edges...")
        for time_index in tqdm(range(total_steps)):
            n_edges = 0
            start_i = int(filtered_data.agent_data.n_agents[time_index])
            existing_edges = []
            for particle_id in monomer_data[time_index]["particles"]:
                particle = monomer_data[time_index]["particles"][particle_id]
                for neighbor_id in particle["neighbor_ids"]:
                    if (particle_id, neighbor_id) in existing_edges:
                        continue
                    neighbor = monomer_data[time_index]["particles"][neighbor_id]
                    positions = np.array([particle["position"], neighbor["position"]])
                    agent_index = start_i + n_edges
                    new_agent_data.unique_ids[time_index][agent_index] = max_used_uid + n_edges
                    new_agent_data.subpoints[time_index][agent_index] = positions
                    existing_edges.append((neighbor_id, particle_id))
                    n_edges += 1
            end_i = start_i + n_edges
            new_agent_data.n_agents[time_index] += n_edges
            new_agent_data.viz_types[time_index][start_i:end_i] = n_edges * [VIZ_TYPE.FIBER]
            new_agent_data.types[time_index] += n_edges * ["edge"]
            new_agent_data.radii[time_index][start_i:end_i] = n_edges * [0.5]
            new_agent_data.n_subpoints[time_index][start_i:end_i] = n_edges * [2.]
        new_agent_data.display_data = {
            "edge" : DisplayData(
                name="edge",
                display_type=DISPLAY_TYPE.FIBER,
                color="#222222",  # gray
            ),
        }
        filtered_data.agent_data = new_agent_data
        return filtered_data
        

    @staticmethod
    def visualize_actin(
        path_to_readdy_h5: str, 
        box_size: np.ndarray, 
        total_steps: int,  
        save_in_one_file: bool,
        file_prefix: str = "",
        flags_to_change: Dict[str, str] = None,
        color: str = "",
        visualize_edges: bool = False,
        monomer_data: List[Dict[str, Any]] = None, 
        plots: List[Dict[str, Any]] = None
    ) -> TrajectoryData:
        """
        visualize an actin trajectory in Simularium
        """
        # rename agents with trajectory suffix if saving in one file
        if save_in_one_file:
            suffix = os.path.basename(path_to_readdy_h5)
            suffix = suffix[suffix.index(file_prefix) + len(file_prefix):]
            suffix = os.path.splitext(suffix)[0]
            suffix = suffix.replace("_", " ")
            suffix = suffix.strip()
            display_data = copy.deepcopy(DISPLAY_DATA)
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
                if len(color) > 0:
                    display_data[agent_type].color = color
        else:
            display_data = DISPLAY_DATA
        # convert
        data = ReaddyData(
            # assume 1e3 recorded steps
            timestep=TIMESTEP * total_steps * 1e-3,
            path_to_readdy_h5=path_to_readdy_h5,
            meta_data=MetaData(
                box_size=box_size,
            ),
            display_data=display_data,
            time_units=UnitData("µs"),
            spatial_units=UnitData("nm"),
        )
        converter = ReaddyConverter(data)
        if plots is not None:
            for plot_type in plots:
                for plot in plots[plot_type]:
                    converter.add_plot(plot, plot_type)
        filtered_data = converter.filter_data([
            MultiplyTimeFilter(
                multiplier=1e-3,
                apply_to_plots=False,
            )
        ])
        if visualize_edges:
            filtered_data = ActinVisualization._add_edge_agents(
                filtered_data, monomer_data
            )
        return filtered_data

    @staticmethod
    def save_actin(
        trajectory_datas: List[TrajectoryData],
        output_path:str, 
        plots: List[Dict[str, Any]] = None
    ):
        """
        save a simularium file with actin trajector(ies)
        """
        traj_data = trajectory_datas[0]
        for index in range(1, len(trajectory_datas)):
            traj_data = TrajectoryConverter(traj_data).filter_data([
                AddAgentsFilter(
                    new_agent_data=trajectory_datas[index].agent_data,
                )
            ])
        if plots is not None:
            traj_data.plots = plots
        BinaryWriter.save(traj_data, output_path)
