#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import (
    MetaData,
    UnitData,
    ScatterPlotData,
    DisplayData,
    DISPLAY_TYPE,
    BinaryWriter,
)
from simulariumio.filters import MultiplyTimeFilter
from simulariumio.orientations import OrientationData, NeighborData
from ..actin import ActinAnalyzer, ACTIN_REACTIONS
from ..common import ReaddyUtil


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
ACTIN_ORIENTATION_DATA = [
    OrientationData(
        type_name_substrings=["actin", "1"],
        neighbor_data=[
            NeighborData(
                type_name_substrings=["actin", "3"],
                relative_position=(
                    np.array([21.847, 24.171, 27.148])
                    - np.array([24.738, 20.881, 26.671])
                ),
                neighbor_type_name_substrings=["actin", "2"],
                neighbor_relative_position=(
                    np.array([19.126, 20.838, 27.757])
                    - np.array([24.738, 20.881, 26.671])
                ),
            ),
            NeighborData(
                type_name_substrings=["actin", "2"],
                relative_position=(
                    np.array([27.609, 24.061, 27.598])
                    - np.array([24.738, 20.881, 26.671])
                ),
                neighbor_type_name_substrings=["actin", "3"],
                neighbor_relative_position=(
                    np.array([30.382, 21.190, 25.725])
                    - np.array([24.738, 20.881, 26.671])
                ),
            ),
        ],
    ),
    OrientationData(
        type_name_substrings=["actin", "2"],
        neighbor_data=[
            NeighborData(
                type_name_substrings=["actin", "1"],
                relative_position=(
                    np.array([21.847, 24.171, 27.148])
                    - np.array([24.738, 20.881, 26.671])
                ),
                neighbor_type_name_substrings=["actin", "3"],
                neighbor_relative_position=(
                    np.array([19.126, 20.838, 27.757])
                    - np.array([24.738, 20.881, 26.671])
                ),
            ),
            NeighborData(
                type_name_substrings=["actin", "3"],
                relative_position=(
                    np.array([27.609, 24.061, 27.598])
                    - np.array([24.738, 20.881, 26.671])
                ),
                neighbor_type_name_substrings=["actin", "1"],
                neighbor_relative_position=(
                    np.array([30.382, 21.190, 25.725])
                    - np.array([24.738, 20.881, 26.671])
                ),
            ),
        ],
    ),
    OrientationData(
        type_name_substrings=["actin", "3"],
        neighbor_data=[
            NeighborData(
                type_name_substrings=["actin", "2"],
                relative_position=(
                    np.array([21.847, 24.171, 27.148])
                    - np.array([24.738, 20.881, 26.671])
                ),
                neighbor_type_name_substrings=["actin", "1"],
                neighbor_relative_position=(
                    np.array([19.126, 20.838, 27.757])
                    - np.array([24.738, 20.881, 26.671])
                ),
            ),
            NeighborData(
                type_name_substrings=["actin", "1"],
                relative_position=(
                    np.array([27.609, 24.061, 27.598])
                    - np.array([24.738, 20.881, 26.671])
                ),
                neighbor_type_name_substrings=["actin", "2"],
                neighbor_relative_position=(
                    np.array([30.382, 21.190, 25.725])
                    - np.array([24.738, 20.881, 26.671])
                ),
            ),
        ],
    ),
    OrientationData(
        type_name_substrings=["actin#branch"],
        neighbor_data=[
            NeighborData(
                type_name_substrings=["arp2#branched"],
                relative_position=(
                    np.array([28.087, 30.872, 26.657])
                    - np.array([29.821, 33.088, 23.356])
                ),
                neighbor_type_name_substrings=["arp3"],
                neighbor_relative_position=(
                    np.array([29.275, 27.535, 23.944])
                    - np.array([29.821, 33.088, 23.356])
                ),
            ),
            NeighborData(
                type_name_substrings=["actin", "2"],
                relative_position=(
                    np.array([30.476, 36.034, 26.528])
                    - np.array([29.821, 33.088, 23.356])
                ),
            ),
        ],
    ),
    OrientationData(
        type_name_substrings=["arp2#free"],
        neighbor_data=[
            NeighborData(
                type_name_substrings=["arp3"],
                relative_position=(
                    np.array([29.275, 27.535, 23.944])
                    - np.array([28.087, 30.872, 26.657])
                ),
                neighbor_type_name_substrings=["actin"],
                neighbor_relative_position=(
                    np.array([30.382, 21.190, 25.725])
                    - np.array([29.275, 27.535, 23.944])
                ),
            ),
            NeighborData(
                type_name_substrings=["actin"],
                relative_position=(
                    np.array([29.821, 33.088, 23.356])
                    - np.array([28.087, 30.872, 26.657])
                ),
            ),
        ],
    ),
    OrientationData(
        type_name_substrings=["arp2#branched"],
        neighbor_data=[
            NeighborData(
                type_name_substrings=["arp3"],
                relative_position=(
                    np.array([29.275, 27.535, 23.944])
                    - np.array([28.087, 30.872, 26.657])
                ),
            ),
            NeighborData(
                type_name_substrings=["actin#branch"],
                relative_position=(
                    np.array([29.821, 33.088, 23.356])
                    - np.array([28.087, 30.872, 26.657])
                ),
                neighbor_type_name_substrings=["actin"],
                neighbor_relative_position=(
                    np.array([30.476, 36.034, 26.528])
                    - np.array([28.087, 30.872, 26.657])
                ),
            ),
        ],
    ),
    OrientationData(
        type_name_substrings=["arp2"],
        neighbor_data=[
            NeighborData(
                type_name_substrings=["arp3"],
                relative_position=(
                    np.array([29.275, 27.535, 23.944])
                    - np.array([28.087, 30.872, 26.657])
                ),
            ),
            NeighborData(
                type_name_substrings=["actin"],
                relative_position=(
                    np.array([27.609, 24.061, 27.598])
                    - np.array([28.087, 30.872, 26.657])
                ),
            ),
        ],
    ),
    OrientationData(
        type_name_substrings=["arp3"],
        neighbor_data=[
            NeighborData(
                type_name_substrings=["arp2"],
                relative_position=(
                    np.array([28.087, 30.872, 26.657])
                    - np.array([29.275, 27.535, 23.944])
                ),
            ),
            NeighborData(
                type_name_substrings=["actin"],
                relative_position=(
                    np.array([30.382, 21.190, 25.725])
                    - np.array([29.275, 27.535, 23.944])
                ),
            ),
        ],
    ),
]

class ActinVisualization:
    """
    visualize an actin trajectory in Simularium
    """

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
        path_to_readdy_h5, box_size, stride=1, periodic_boundary=True, plots=None
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
        (
            monomer_data,
            reactions,
            times,
            _,
        ) = ReaddyUtil.monomer_data_and_reactions_from_file(
            h5_file_path=path_to_readdy_h5,
            stride=stride,
            timestep=0.1,
            reaction_names=ACTIN_REACTIONS,
        )
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
        return ScatterPlotData(
            title="Twist along filament",
            xaxis_title="Pointed end displacement (nm)",
            yaxis_title="Twist (rotations)",
            xtrace=ActinAnalyzer.analyze_pointed_end_displacement(
                monomer_data, box_size, periodic_boundary
            ),
            ytraces={
                "Total twist (degrees)": ActinAnalyzer.analyze_total_twist(
                    monomer_data, box_size, periodic_boundary, remove_bend=False
                ),
                "Total twist excluding bend (degrees)": (
                    ActinAnalyzer.analyze_total_twist(
                        monomer_data, box_size, periodic_boundary, remove_bend=True
                    )
                ),
            },
            render_mode="lines",
        )

    @staticmethod
    def get_bond_length_plot(monomer_data, box_size, periodic_boundary, times):
        """
        Add a plot of bond lengths (lat and long) vs end displacement
        (normalize bond lengths relative to theoretical lengths, plot average ± std)
        """
        lateral_bond_lengths = ActinAnalyzer.analyze_lateral_bond_lengths(
            monomer_data, box_size, periodic_boundary
        )
        mean_lat = ActinAnalyzer.analyze_average_for_series(lateral_bond_lengths)
        stddev_lat = ActinAnalyzer.analyze_stddev_for_series(lateral_bond_lengths)
        longitudinal_bond_lengths = ActinAnalyzer.analyze_longitudinal_bond_lengths(
            monomer_data, box_size, periodic_boundary
        )
        mean_long = ActinAnalyzer.analyze_average_for_series(longitudinal_bond_lengths)
        stddev_long = ActinAnalyzer.analyze_stddev_for_series(longitudinal_bond_lengths)
        return ScatterPlotData(
            title="Bond lengths",
            xaxis_title="Pointed end displacement (nm)",
            yaxis_title="Normalized bond length",
            xtrace=ActinAnalyzer.analyze_pointed_end_displacement(
                monomer_data, box_size, periodic_boundary
            ),
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
    def generate_bend_twist_plots(
        path_to_readdy_h5, box_size, stride=1, periodic_boundary=True, plots=None
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
        (
            monomer_data,
            reactions,
            times,
            _,
        ) = ReaddyUtil.monomer_data_and_reactions_from_file(
            h5_file_path=path_to_readdy_h5,
            stride=stride,
            timestep=0.1,
            reaction_names=ACTIN_REACTIONS,
        )
        plots["scatter"] += [
            ActinVisualization.get_total_twist_plot(
                monomer_data, box_size, periodic_boundary, times
            ),
            ActinVisualization.get_bond_length_plot(
                monomer_data, box_size, periodic_boundary, times
            ),
        ]
        return plots

    @staticmethod
    def visualize_actin(path_to_readdy_h5, box_size, total_steps, plots=None):
        """
        visualize an actin trajectory in Simularium
        """
        # radii
        extra_radius = 1.5
        actin_radius = 2.0 + extra_radius
        arp23_radius = 2.0 + extra_radius
        cap_radius = 3.0 + extra_radius
        obstacle_radius = 35.0
        bucket_url = "https://aics-simularium-data.s3.us-east-2.amazonaws.com"
        # test_geo = f"{bucket_url}/geometry/gizmo_small.obj"
        arp2_geometry_url = f"{bucket_url}/geometry/arp2.pdb"
        arp3_geometry_url = f"{bucket_url}/geometry/arp3.pdb"
        actin_display_type = DISPLAY_TYPE.PDB
        actin_geometry_url = f"{bucket_url}/geometry/actin.pdb"
        actin_branch_geometry_url= f"{bucket_url}/geometry/actin#branch.pdb"
        arp_display_type = DISPLAY_TYPE.PDB
        colors = {
            "arp2" : "#74d4e3",
            "arp3" : "#77ab59",
            "arp3#ATP" : "#99ff66",
            "cap" : "#005073",
            "cap#bound" : "#189ad3",
            "actin#barbed" : "#ffff00",
            "actin#ATP" : "#ffcc33",
            "actin" : "#fc9f32",
            "actin#pointed" : "#f25622",
            "obstacle" : "#666666",
        }
        display_data = {
            "arp2": DisplayData(
                name="arp2",
                radius=arp23_radius,
                display_type=arp_display_type,
                url=arp2_geometry_url,
                color=colors["arp2"],
            ),
            "arp2#branched": DisplayData(
                name="arp2#branched",
                radius=arp23_radius,
                display_type=arp_display_type,
                url=arp2_geometry_url,
                color=colors["arp2"],
            ),
            "arp2#free": DisplayData(
                name="arp2#free",
                radius=arp23_radius,
                display_type=arp_display_type,
                url=arp2_geometry_url,
                color=colors["arp2"],
            ),
            "arp3": DisplayData(
                name="arp3",
                radius=arp23_radius,
                display_type=arp_display_type,
                url=arp3_geometry_url,
                color=colors["arp3"],
            ),
            "arp3#new": DisplayData(
                name="arp3",
                radius=arp23_radius,
                display_type=arp_display_type,
                url=arp3_geometry_url,
                color=colors["arp3"],
            ),
            "arp3#ATP": DisplayData(
                name="arp3#ATP",
                radius=arp23_radius,
                display_type=arp_display_type,
                url=arp3_geometry_url,
                color=colors["arp3#ATP"],
            ),
            "arp3#new_ATP": DisplayData(
                name="arp3#ATP",
                radius=arp23_radius,
                display_type=arp_display_type,
                url=arp3_geometry_url,
                color=colors["arp3#ATP"],
            ),
            "cap": DisplayData(
                name="cap",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=cap_radius,
                color=colors["cap"],
            ),
            "cap#new": DisplayData(
                name="cap",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=cap_radius,
                color=colors["cap#bound"],
            ),
            "cap#bound": DisplayData(
                name="cap#bound",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=cap_radius,
                color=colors["cap#bound"],
            ),
            "actin#free": DisplayData(
                name="actin#free",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#free_ATP": DisplayData(
                name="actin#free_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#new": DisplayData(
                name="actin",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#new_ATP": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#1": DisplayData(
                name="actin",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#2": DisplayData(
                name="actin",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#3": DisplayData(
                name="actin",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#ATP_1": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#ATP_2": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#ATP_3": DisplayData(
                name="actin#ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#mid_1": DisplayData(
                name="actin#mid",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#mid_2": DisplayData(
                name="actin#mid",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#mid_3": DisplayData(
                name="actin#mid",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#mid_ATP_1": DisplayData(
                name="actin#mid_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#mid_ATP_2": DisplayData(
                name="actin#mid_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#mid_ATP_3": DisplayData(
                name="actin#mid_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#pointed_1": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_2": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_3": DisplayData(
                name="actin#pointed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_ATP_1": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_ATP_2": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_ATP_3": DisplayData(
                name="actin#pointed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#barbed_1": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#barbed_2": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#barbed_3": DisplayData(
                name="actin#barbed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#barbed_ATP_1": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#barbed_ATP_2": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#barbed_ATP_3": DisplayData(
                name="actin#barbed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#fixed_1": DisplayData(
                name="actin#fixed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#fixed_2": DisplayData(
                name="actin#fixed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#fixed_3": DisplayData(
                name="actin#fixed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#fixed_ATP_1": DisplayData(
                name="actin#fixed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#fixed_ATP_2": DisplayData(
                name="actin#fixed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#fixed_ATP_3": DisplayData(
                name="actin#fixed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#mid_fixed_1": DisplayData(
                name="actin#mid_fixed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#mid_fixed_2": DisplayData(
                name="actin#mid_fixed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#mid_fixed_3": DisplayData(
                name="actin#mid_fixed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin"],
            ),
            "actin#mid_fixed_ATP_1": DisplayData(
                name="actin#mid_fixed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#mid_fixed_ATP_2": DisplayData(
                name="actin#mid_fixed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#mid_fixed_ATP_3": DisplayData(
                name="actin#mid_fixed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#pointed_fixed_1": DisplayData(
                name="actin#pointed_fixed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_fixed_2": DisplayData(
                name="actin#pointed_fixed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_fixed_3": DisplayData(
                name="actin#pointed_fixed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_fixed_ATP_1": DisplayData(
                name="actin#pointed_fixed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_fixed_ATP_2": DisplayData(
                name="actin#pointed_fixed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#pointed_fixed_ATP_3": DisplayData(
                name="actin#pointed_fixed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#pointed"],
            ),
            "actin#fixed_barbed_1": DisplayData(
                name="actin#fixed_barbed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#fixed_barbed_2": DisplayData(
                name="actin#fixed_barbed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#fixed_barbed_3": DisplayData(
                name="actin#fixed_barbed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#fixed_barbed_ATP_1": DisplayData(
                name="actin#fixed_barbed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#fixed_barbed_ATP_2": DisplayData(
                name="actin#fixed_barbed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#fixed_barbed_ATP_3": DisplayData(
                name="actin#fixed_barbed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#branch_1": DisplayData(
                name="actin#branch",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_branch_geometry_url,
                color=colors["actin"],
            ),
            "actin#branch_ATP_1": DisplayData(
                name="actin#branch_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_branch_geometry_url,
                color=colors["actin#ATP"],
            ),
            "actin#branch_barbed_1": DisplayData(
                name="actin#branch_barbed",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_branch_geometry_url,
                color=colors["actin#barbed"],
            ),
            "actin#branch_barbed_ATP_1": DisplayData(
                name="actin#branch_barbed_ATP",
                radius=actin_radius,
                display_type=actin_display_type,
                url=actin_branch_geometry_url,
                color=colors["actin#barbed"],
            ),
            "obstacle": DisplayData(
                name="obstacle",
                display_type=DISPLAY_TYPE.SPHERE,
                radius=obstacle_radius,
                color=colors["obstacle"],
            ),
        }
        # convert
        data = ReaddyData(
            # assume 1e3 recorded steps
            timestep=TIMESTEP * total_steps * 1e-3,
            path_to_readdy_h5=path_to_readdy_h5,
            meta_data=MetaData(
                box_size=np.array(3 *[box_size]),
            ),
            display_data=display_data,
            zero_orientations=ACTIN_ORIENTATION_DATA,
            time_units=UnitData("µs"),
            spatial_units=UnitData("nm"),
            plots=[],
        )
        try:
            converter = ReaddyConverter(data)
        except OverflowError as e:
            print(
                "OverflowError during SimulariumIO conversion !!!!!!!!!!!!!!\n" + str(e)
            )
            return
        if plots is not None:
            for plot_type in plots:
                for plot in plots[plot_type]:
                    converter.add_plot(plot, plot_type)
        filtered_data = converter.filter_data(
            [
                MultiplyTimeFilter(
                    multiplier=1e-3,
                    apply_to_plots=False,
                ),
            ]
        )
        BinaryWriter.save(filtered_data, path_to_readdy_h5)
