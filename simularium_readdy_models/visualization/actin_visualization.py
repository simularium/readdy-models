#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Any

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import (
    TrajectoryConverter,
    MetaData,
    UnitData,
    ScatterPlotData,
    DisplayData,
    DISPLAY_TYPE,
    TrajectoryData,
    DimensionData,
    CameraData,
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

extra_radius = 0.0
actin_radius = 1.0  # 2.0 + extra_radius
arp23_radius = 2.0 + extra_radius
cap_radius = 3.0 + extra_radius
obstacle_radius = 35.0

ACTIN_DISPLAY_DATA = {
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
        h5_file_path,
        stride=1,
        reactions=False,
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
                            monomer_data,
                            box_size,
                            periodic_boundary,
                        )
                    )
                ),
            },
            render_mode="lines",
        )

    @staticmethod
    def generate_polymerization_plots(
        monomer_data,
        times,
        reactions,
        box_size,
        periodic_boundary=True,
        plots=None,
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
    def get_total_axis_twist_plot(axis_twist, times):
        """
        Add a plot of total axis twist vs time
        """
        axis_sum = np.sum(axis_twist, axis=1)
        return ScatterPlotData(
            title="Total filament axis twist",
            xaxis_title="T (μs)",
            yaxis_title="Twist (rotations)",
            xtrace=times,
            ytraces={
                "<<<": 9.5 * np.ones(axis_sum.shape),
                ">>>": 15.5 * np.ones(axis_sum.shape),
                "Ideal": axis_sum[0] * np.ones(axis_sum.shape),
                "Axis angle": axis_sum,
            },
            render_mode="lines",
        )

    @staticmethod
    def get_total_plane_twist_plot(plane_twist, times):
        """
        Add a plot of total plane twist vs time
        """
        plane_sum = np.sum(plane_twist, axis=1)
        plane_sum /= plane_sum[0]
        return ScatterPlotData(
            title="Total filament plane twist",
            xaxis_title="T (μs)",
            yaxis_title="Twist (normalized)",
            xtrace=times,
            ytraces={
                "Total": plane_sum,
            },
            render_mode="lines",
        )

    @staticmethod
    def get_twist_per_monomer_plot(twist_angles, filament_positions):
        """
        Add a plot of twist vs position of the monomer in filament
        """
        end_time = len(twist_angles) - 1
        return ScatterPlotData(
            title="Final twist along filament",
            xaxis_title="Filament position (index)",
            yaxis_title="Twist (rotations)",
            xtrace=filament_positions[end_time],
            ytraces={
                "End": twist_angles[end_time],
            },
            render_mode="lines",
        )

    @staticmethod
    def get_total_bond_energy_plot(
        lateral_bond_energies, longitudinal_bond_energies, times
    ):
        """
        Add a plot of bond energies (lat and long) vs time
        """
        sum_lat = np.sum(lateral_bond_energies, axis=1)
        sum_long = np.sum(longitudinal_bond_energies, axis=1)
        return ScatterPlotData(
            title="Total bond energy",
            xaxis_title="T (μs)",
            yaxis_title="Strain energy (KT)",
            xtrace=times,
            ytraces={
                "Lateral": sum_lat,
                "Longitudinal": sum_long,
            },
            render_mode="lines",
        )

    @staticmethod
    def get_bond_energy_per_monomer_plot(
        lateral_bond_energies, longitudinal_bond_energies, filament_positions
    ):
        """
        Add a plot of bond energies (lat and long) vs index of monomer in filament
        """
        end_time = len(lateral_bond_energies) - 1
        return ScatterPlotData(
            title="Final bond energy along filament",
            xaxis_title="Filament position (index)",
            yaxis_title="Strain Energy (KT)",
            xtrace=filament_positions[end_time],
            ytraces={
                "Lateral": lateral_bond_energies[end_time],
                "Longitudinal": longitudinal_bond_energies[end_time],
            },
            render_mode="lines",
        )

    @staticmethod
    def get_filament_length_plot(
        normals, axis_positions, box_size, periodic_boundary, stride, times
    ):
        """
        Add a plot of distance from first to last particle
        in the first filament vs time
        """
        filament_length = ActinAnalyzer.analyze_filament_length(
            normals, axis_positions, box_size, periodic_boundary, stride
        )
        return ScatterPlotData(
            title="Filament length",
            xaxis_title="T (μs)",
            yaxis_title="Length of filament (nm)",
            xtrace=times[::stride],
            ytraces={
                "<<<": 450.0 * np.ones(filament_length.shape),
                ">>>": 550.0 * np.ones(filament_length.shape),
                "Ideal": filament_length[0] * np.ones(filament_length.shape),
                "Filament length": filament_length,
            },
            render_mode="lines",
        )

    @staticmethod
    def get_bond_stretch_plot(monomer_data, box_size, periodic_boundary, stride, times):
        """
        Add a scatter plot of difference in bond length from ideal
        for lateral and longitudinal actin bonds vs time
        """
        (
            stretch_lat,
            stretch_long,
        ) = ActinAnalyzer.analyze_bond_stretch(
            monomer_data, box_size, periodic_boundary, stride
        )
        mean_lat = np.mean(stretch_lat, axis=1)
        # stddev_lat = np.std(stretch_lat, axis=1)
        mean_long = np.mean(stretch_long, axis=1)
        # stddev_long = np.std(stretch_long, axis=1)
        return ScatterPlotData(
            title="Bond stretch",
            xaxis_title="T (μs)",
            yaxis_title="Bond stretch (nm)",
            xtrace=times[::stride],
            ytraces={
                "<<<": -1.0 * np.ones(mean_lat.shape),
                ">>>": 1.0 * np.ones(mean_lat.shape),
                "Lateral mean": mean_lat,
                "Longitudinal mean": mean_long,
            },
            render_mode="lines",
        )

    @staticmethod
    def get_angle_stretch_plot(
        monomer_data, box_size, periodic_boundary, stride, times
    ):
        """
        Add a scatter plot of difference in angles from ideal
        for lateral and longitudinal actin bonds vs time
        """
        (
            stretch_lat_lat,
            stretch_lat_long,
            stretch_long_long,
        ) = ActinAnalyzer.analyze_angle_stretch(
            monomer_data, box_size, periodic_boundary, stride
        )
        mean_lat_lat = np.mean(stretch_lat_lat, axis=1)
        mean_lat_long = np.mean(stretch_lat_long, axis=1)
        mean_long_long = np.mean(stretch_long_long, axis=1)
        return ScatterPlotData(
            title="Angle stretch",
            xaxis_title="T (μs)",
            yaxis_title="Angle stretch (degrees)",
            xtrace=times[::stride],
            ytraces={
                "<<<": -10.0 * np.ones(mean_lat_lat.shape),
                ">>>": 10.0 * np.ones(mean_lat_lat.shape),
                "Lat to lat mean": mean_lat_lat,
                "Lat to long mean": mean_lat_long,
                "Long to long mean": mean_long_long,
            },
            render_mode="lines",
        )

    @staticmethod
    def get_dihedral_stretch_plot(
        monomer_data, box_size, periodic_boundary, stride, times
    ):
        """
        Add a scatter plot of difference in angles from ideal
        for lateral and longitudinal actin bonds vs time
        """
        (
            stretch_lat_lat_lat,
            stretch_long_long_long,
        ) = ActinAnalyzer.analyze_dihedral_stretch(
            monomer_data, box_size, periodic_boundary, stride
        )
        mean_lat_lat_lat = np.mean(stretch_lat_lat_lat, axis=1)
        mean_long_long_long = np.mean(stretch_long_long_long, axis=1)
        return ScatterPlotData(
            title="Dihedral stretch",
            xaxis_title="T (μs)",
            yaxis_title="Angle stretch (degrees)",
            xtrace=times[::stride],
            ytraces={
                "<<<": -10.0 * np.ones(mean_lat_lat_lat.shape),
                ">>>": 10.0 * np.ones(mean_lat_lat_lat.shape),
                "Lat to lat to lat mean": mean_lat_lat_lat,
                "Long to long to long mean": mean_long_long_long,
            },
            render_mode="lines",
        )

    @staticmethod
    def generate_filament_structure_plots(
        monomer_data,
        times,
        box_size,
        normals,
        axis_positions,
        periodic_boundary=True,
        plots=None,
    ):
        """
        Use an ActinAnalyzer to generate plots of observables
        for a diffusing actin filament
        """
        STRIDE = 10
        if plots is None:
            plots = {
                "scatter": [],
                "histogram": [],
            }
        axis_twist, _, _ = ActinAnalyzer.analyze_twist_axis(
            normals, axis_positions, STRIDE
        )
        plots["scatter"] += [
            ActinVisualization.get_total_axis_twist_plot(axis_twist, times[::STRIDE]),
            ActinVisualization.get_filament_length_plot(
                normals, axis_positions, box_size, periodic_boundary, STRIDE, times
            ),
            ActinVisualization.get_bond_stretch_plot(
                monomer_data, box_size, periodic_boundary, STRIDE, times
            ),
            ActinVisualization.get_angle_stretch_plot(
                monomer_data, box_size, periodic_boundary, STRIDE, times
            ),
            ActinVisualization.get_dihedral_stretch_plot(
                monomer_data, box_size, periodic_boundary, STRIDE, times
            ),
        ]
        return plots

    @staticmethod
    def generate_bend_twist_plots(
        monomer_data,
        times,
        box_size,
        normals,
        axis_positions,
        periodic_boundary=True,
        plots=None,
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
        STRIDE = 10
        (axis_twist, _, _) = ActinAnalyzer.analyze_twist_axis(
            normals, axis_positions, STRIDE
        )
        (twist_angles, filament_positions1) = ActinAnalyzer.analyze_twist_planes(
            monomer_data, box_size, periodic_boundary, STRIDE
        )
        (
            lateral_bond_energies,
            longitudinal_bond_energies,
            filament_positions2,
        ) = ActinAnalyzer.analyze_bond_energies(
            monomer_data, box_size, periodic_boundary, STRIDE
        )
        plots["scatter"] += [
            ActinVisualization.get_total_axis_twist_plot(axis_twist, times[::STRIDE]),
            ActinVisualization.get_total_plane_twist_plot(
                twist_angles, times[::STRIDE]
            ),
            ActinVisualization.get_twist_per_monomer_plot(
                twist_angles, filament_positions1
            ),
            ActinVisualization.get_total_bond_energy_plot(
                lateral_bond_energies, longitudinal_bond_energies, times[::STRIDE]
            ),
            ActinVisualization.get_bond_energy_per_monomer_plot(
                lateral_bond_energies, longitudinal_bond_energies, filament_positions2
            ),
        ]
        return plots

    @staticmethod
    def _get_added_dimensions_for_lines(
        traj_data: TrajectoryData,
        max_agents: int,
    ) -> DimensionData:
        """
        Get a DimensionData with the deltas for each dimension
        of AgentData when adding fibers with 2 points each
        """
        current_dimensions = traj_data.agent_data.get_dimensions()
        return DimensionData(
            total_steps=0,
            max_agents=max_agents,
            max_subpoints=6 - current_dimensions.max_subpoints,
        )

    @staticmethod
    def _add_normal_agents(
        traj_data: TrajectoryData,
        monomer_data: List[Dict[str, Any]],
        normals: List[List[Any]],
        axis_positions: List[List[Any]],
        type_name: str,
        color: str = "",
    ) -> TrajectoryData:
        """
        Add agent data for fibers to draw normals for each actin in a filament
        """
        if monomer_data is None:
            raise Exception("Normal visualization requires monomer_data")
        # get dimensions of data
        total_steps = len(monomer_data)
        max_normals = 0
        for time_i in range(total_steps):
            n_normals = len(normals[time_i])
            if n_normals > max_normals:
                max_normals = n_normals
        dimensions = ActinVisualization._get_added_dimensions_for_lines(
            traj_data, max_normals
        )
        new_agent_data = traj_data.agent_data.get_copy_with_increased_buffer_size(
            dimensions
        )
        new_type_name = f"{type_name}#normal" if type_name else "normal"
        # add new agents
        print("Processing normals...")
        max_used_uid = max(list(np.unique(traj_data.agent_data.unique_ids)))
        for time_i in range(total_steps):
            start_i = int(traj_data.agent_data.n_agents[time_i])
            max_normals = len(normals[time_i])
            n_normals = 0
            for normal_i in range(max_normals):
                if (
                    axis_positions[time_i][normal_i] is None
                    or normals[time_i][normal_i] is None
                ):
                    continue
                agent_i = start_i + n_normals
                new_agent_data.unique_ids[time_i][agent_i] = max_used_uid + normal_i + 1
                new_agent_data.subpoints[time_i][agent_i] = np.array(
                    [
                        axis_positions[time_i][normal_i],
                        axis_positions[time_i][normal_i]
                        + 10 * normals[time_i][normal_i],
                    ]
                ).flatten()
                n_normals += 1
            end_i = start_i + n_normals
            new_agent_data.n_agents[time_i] += n_normals
            new_agent_data.viz_types[time_i][start_i:end_i] = n_normals * [
                VIZ_TYPE.FIBER
            ]
            new_agent_data.types[time_i] += n_normals * [new_type_name]
            new_agent_data.radii[time_i][start_i:end_i] = n_normals * [0.5]
            new_agent_data.n_subpoints[time_i][start_i:end_i] = n_normals * [6.0]
        new_agent_data.display_data[new_type_name] = DisplayData(
            name=new_type_name,
            display_type=DISPLAY_TYPE.FIBER,
            color="#685bf3" if not color else color,  # default to blue
        )
        traj_data.agent_data = new_agent_data
        return traj_data

    @staticmethod
    def _add_edge_agents(
        traj_data: TrajectoryData,
        monomer_data: List[Dict[str, Any]],
        type_name: str,
        color: str = "",
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
        dimensions = ActinVisualization._get_added_dimensions_for_lines(
            traj_data, max_edges
        )
        new_agent_data = traj_data.agent_data.get_copy_with_increased_buffer_size(
            dimensions
        )
        new_type_name = f"{type_name}#edge" if type_name else "edge"
        # add new agents
        print("Processing edges...")
        max_used_uid = max(list(np.unique(traj_data.agent_data.unique_ids)))
        for time_index in tqdm(range(total_steps)):
            n_edges = 0
            start_i = int(traj_data.agent_data.n_agents[time_index])
            existing_edges = []
            for particle_id in monomer_data[time_index]["particles"]:
                particle = monomer_data[time_index]["particles"][particle_id]
                for neighbor_id in particle["neighbor_ids"]:
                    if (particle_id, neighbor_id) in existing_edges:
                        continue
                    neighbor = monomer_data[time_index]["particles"][neighbor_id]
                    positions = np.array(
                        [particle["position"], neighbor["position"]]
                    ).flatten()
                    agent_index = start_i + n_edges
                    new_agent_data.unique_ids[time_index][agent_index] = (
                        max_used_uid + n_edges
                    )
                    new_agent_data.subpoints[time_index][agent_index] = positions
                    existing_edges.append((neighbor_id, particle_id))
                    n_edges += 1
            end_i = start_i + n_edges
            new_agent_data.n_agents[time_index] += n_edges
            new_agent_data.viz_types[time_index][start_i:end_i] = n_edges * [
                VIZ_TYPE.FIBER
            ]
            new_agent_data.types[time_index] += n_edges * [new_type_name]
            new_agent_data.radii[time_index][start_i:end_i] = n_edges * [0.5]
            new_agent_data.n_subpoints[time_index][start_i:end_i] = n_edges * [6.0]
        new_agent_data.display_data[new_type_name] = DisplayData(
            name=new_type_name,
            display_type=DISPLAY_TYPE.FIBER,
            color="#eaeaea" if not color else color,  # default to light gray
        )
        traj_data.agent_data = new_agent_data
        return traj_data

    @staticmethod
    def visualize_actin(
        path_to_readdy_h5: str,
        box_size: np.ndarray,
        total_steps: int,
        suffix: str = "",
        display_data: Dict[str, DisplayData] = None,
        color: str = "",
        visualize_edges: bool = False,
        visualize_normals: bool = False,
        monomer_data: List[Dict[str, Any]] = None,
        normals: List[List[Any]] = None,
        axis_positions: List[List[Any]] = None,
        plots: List[Dict[str, Any]] = None,
        longitudinal_bonds: bool = True,
    ) -> TrajectoryData:
        """
        visualize an actin trajectory in Simularium
        """
        # radii
        extra_radius = 1.5
        actin_radius = 2.0 + extra_radius
        arp23_radius = 2.0 + extra_radius
        cap_radius = 3.0 + extra_radius
        obstacle_radius = 35.0
        display_data = {
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
        n_polymer_numbers = 3 if not longitudinal_bonds else 5
        for i in range(1, n_polymer_numbers + 1):
            display_data.update(
                {
                    f"actin#{i}": DisplayData(
                        name="actin",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#bf9b30",
                    ),
                    f"actin#mid_{i}": DisplayData(
                        name="actin#mid",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#bf9b30",
                    ),
                    f"actin#fixed_{i}": DisplayData(
                        name="actin#fixed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#bf9b30",
                    ),
                    f"actin#mid_fixed_{i}": DisplayData(
                        name="actin#mid_fixed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#bf9b30",
                    ),
                    f"actin#ATP_{i}": DisplayData(
                        name="actin#ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffbf00",
                    ),
                    f"actin#mid_ATP_{i}": DisplayData(
                        name="actin#mid_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffbf00",
                    ),
                    f"actin#fixed_ATP_{i}": DisplayData(
                        name="actin#fixed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffbf00",
                    ),
                    f"actin#mid_fixed_ATP_{i}": DisplayData(
                        name="actin#mid_fixed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffbf00",
                    ),
                    f"actin#barbed_{i}": DisplayData(
                        name="actin#barbed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffdc73",
                    ),
                    f"actin#barbed_ATP_{i}": DisplayData(
                        name="actin#barbed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffdc73",
                    ),
                    f"actin#fixed_barbed_{i}": DisplayData(
                        name="actin#fixed_barbed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffdc73",
                    ),
                    f"actin#fixed_barbed_ATP_{i}": DisplayData(
                        name="actin#fixed_barbed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#ffdc73",
                    ),
                    f"actin#pointed_{i}": DisplayData(
                        name="actin#pointed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#a67c00",
                    ),
                    f"actin#pointed_ATP_{i}": DisplayData(
                        name="actin#pointed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#a67c00",
                    ),
                    f"actin#pointed_fixed_{i}": DisplayData(
                        name="actin#pointed_fixed",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#a67c00",
                    ),
                    f"actin#pointed_fixed_ATP_{i}": DisplayData(
                        name="actin#pointed_fixed_ATP",
                        display_type=DISPLAY_TYPE.SPHERE,
                        radius=actin_radius,
                        color="#a67c00",
                    ),
                },
            )
        # convert
        data = ReaddyData(
            # assume 1e3 recorded steps
            timestep=TIMESTEP * total_steps * 1e-3,
            path_to_readdy_h5=path_to_readdy_h5,
            meta_data=MetaData(
                box_size=box_size,
                camera_defaults=CameraData(
                    position=np.array([0.0, 0.0, 300.0]),
                    look_at_position=np.zeros(3),
                    up_vector=np.array([0.0, 1.0, 0.0]),
                    fov_degrees=120.0,
                ),
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
        filtered_data = converter.filter_data(
            [
                MultiplyTimeFilter(
                    multiplier=1e-3,
                    apply_to_plots=False,
                )
            ]
        )
        if visualize_edges:
            filtered_data = ActinVisualization._add_edge_agents(
                filtered_data, monomer_data, suffix, color
            )
        if visualize_normals:
            filtered_data = ActinVisualization._add_normal_agents(
                filtered_data, monomer_data, normals, axis_positions, suffix, color
            )
        return filtered_data

    @staticmethod
    def save_actin(
        trajectory_datas: List[TrajectoryData],
        output_path: str,
        plots: List[Dict[str, Any]] = None,
    ):
        """
        save a simularium file with actin trajector(ies)
        """
        traj_data = trajectory_datas[0]
        for index in range(1, len(trajectory_datas)):
            traj_data = TrajectoryConverter(traj_data).filter_data(
                [
                    AddAgentsFilter(
                        new_agent_data=trajectory_datas[index].agent_data,
                    )
                ]
            )
        converter = TrajectoryConverter(traj_data)
        if plots is not None:
            for plot_type in plots:
                for plot in plots[plot_type]:
                    converter.add_plot(plot, plot_type)
        converter.save(output_path)
