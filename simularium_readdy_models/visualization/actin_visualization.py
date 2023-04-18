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
    CameraData,
)
from simulariumio.plot_readers import ScatterPlotReader
from simulariumio.filters import AddAgentsFilter, MultiplyTimeFilter
from subcell_analysis.readdy import ReaddyPostProcessor
from subcell_analysis import (
    SpatialAnnotator
)

from ..actin import ActinAnalyzer, ACTIN_REACTIONS
from ..common import ReaddyUtil


class ActinVisualization:
    """
    Actin-specific visualization functions to create plots, 
    annotate spatial data, and save simularium files.
    """
    
    TIMESTEP: float = 0.1
    
    @staticmethod
    def ACTIN_DISPLAY_DATA(longitudinal_bonds: bool) -> Dict[str,DisplayData]:
        # radii
        extra_radius = 1.5
        actin_radius = 2.0 + extra_radius
        arp23_radius = 2.0 + extra_radius
        cap_radius = 3.0 + extra_radius
        obstacle_radius = 35.0
        # non-polymer types
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
        # polymer types
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
        return display_data

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
        for total_rxn_name in ACTIN_REACTIONS.GROWTH_RXNS:
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
        for total_rxn_name in ACTIN_REACTIONS.STRUCTURAL_RXNS:
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
        for rxn_group_name in ACTIN_REACTIONS.GROUPED_GROWTH_RXNS:
            group_reaction_events = []
            for total_rxn_name in ACTIN_REACTIONS.GROUPED_GROWTH_RXNS[rxn_group_name]:
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
    def generate_actin_compression_plots(
        post_processor: ReaddyPostProcessor,
        fiber_chain_ids: List[List[List[int]]],
        temperature_c: float,
        plots: List[ScatterPlotData] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use an ActinAnalyzer to generate plots of observables
        for actin being compressed
        """
        if plots is None:
            plots = []
        times = []
        for frame in post_processor.trajectory:
            times.append(frame.time)
        bond_energies = post_processor.fiber_bond_energies(
            fiber_chain_ids=fiber_chain_ids,
            ideal_lengths=ActinAnalyzer.ideal_bond_lengths(),
            k=ActinAnalyzer.bond_energy_constants(
                temp_c=temperature_c
            ),
            stride=10,
        )
        sum_lat = np.sum(bond_energies[1], axis=1)
        sum_long = np.sum(bond_energies[2], axis=1)
        plots.append(
            ScatterPlotData(
                title="Total bond energy",
                xaxis_title="T (μs)",
                yaxis_title="Strain energy (KT)",
                xtrace=times[::10],
                ytraces={
                    "Lateral": sum_lat,
                    "Longitudinal": sum_long,
                },
                render_mode="lines",
            )
        )
        formatted_plots = []
        for plot_type in plots:
            for plot in plots[plot_type]:
                formatted_plots.append(ScatterPlotReader.read(plot))
        return formatted_plots

    @staticmethod
    def add_spatial_annotations(
        traj_data: TrajectoryData,
        post_processor: ReaddyPostProcessor,
        visualize_edges: bool,
        visualize_normals: bool,
        visualize_control_pts: bool,
        fiber_chain_ids: List[List[List[int]]] = None,
        axis_positions: List[List[np.ndarray]] = None,
    ) -> TrajectoryData:
        """
        Use an ActinAnalyzer to generate plots of observables
        for actin being compressed
        """
        if visualize_edges:
            edges = post_processor.edge_positions()
            traj_data = SpatialAnnotator.add_fiber_agents(
                traj_data=traj_data,
                fiber_points=edges,
                type_name="edge",
                fiber_width=0.5,
                color="#eaeaea",
            )
        if visualize_normals:
            if fiber_chain_ids is None:
                raise Exception(
                    "In add_spatial_annotations(), fiber_chain_ids "
                    "are required to visualize normals."
                )
            if axis_positions is None:
                raise Exception(
                    "In add_spatial_annotations(), axis_positions "
                    "are required to visualize normals."
                )
            normals = post_processor.linear_fiber_normals(
                fiber_chain_ids=fiber_chain_ids,
                axis_positions=axis_positions,
                normal_length=5.,
            )
            traj_data = SpatialAnnotator.add_fiber_agents(
                traj_data=traj_data,
                fiber_points=normals,
                type_name="normal",
                fiber_width=0.5,
                color="#685bf3",
            )
        if visualize_control_pts:
            if axis_positions is None:
                raise Exception(
                    "In add_spatial_annotations(), axis_positions "
                    "are required to visualize control points."
                )
            control_points = post_processor.linear_fiber_control_points(
                axis_positions=axis_positions,
                segment_length=0.1,
            )
            traj_data = SpatialAnnotator.add_sphere_agents(
                traj_data=traj_data,
                sphere_positions=control_points,
                type_name="fiber point",
                radius=1.,
                color="#eaeaea",
            )
        return traj_data

    @staticmethod
    def simularium_trajectory(
        path_to_readdy_h5: str,
        box_size: np.ndarray,
        total_steps: int,
        time_multiplier: float,
        longitudinal_bonds: bool = True,
    ) -> TrajectoryData:
        """
        Get a TrajectoryData to visualize an actin trajectory in Simularium
        """
        data = ReaddyData(
            timestep=(
                ActinVisualization.TIMESTEP * total_steps
                * time_multiplier
            ),
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
            display_data=ActinVisualization.ACTIN_DISPLAY_DATA(longitudinal_bonds),
            time_units=UnitData("µs"),
            spatial_units=UnitData("nm"),
        )
        converter = ReaddyConverter(data)
        traj_data = converter.filter_data(
            [
                MultiplyTimeFilter(
                    multiplier=time_multiplier,
                    apply_to_plots=False,
                )
            ]
        )
        return traj_data

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
