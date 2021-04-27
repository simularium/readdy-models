#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio import MetaData, UnitData, ScatterPlotData
from ..actin import ActinAnalyzer


class ActinVisualization:
    """
    visualize an actin trajectory in Simularium
    """

    @staticmethod
    def get_bound_actin_plot(analyzer):
        """
        Add a plot of percent actin in filaments
        """
        return ScatterPlotData(
            title="Filamentous actin",
            xaxis_title="Time (ns)",
            yaxis_title="Bound actin (%)",
            xtrace=analyzer.times,
            ytraces={
                "Bound actin": 100.0
                * np.array(analyzer.analyze_ratio_of_filamentous_to_total_actin()),
            },
            render_mode="lines",
        )

    @staticmethod
    def generate_plots(path_to_readdy_h5, box_size):
        """
        Use an ActinAnalyzer to generate plots of observables
        """
        analyzer = ActinAnalyzer(path_to_readdy_h5, box_size)
        return [
            ActinVisualization.get_bound_actin_plot(analyzer),
        ]
        # ATP_actin_ratio = analyzer.analyze_ratio_of_ATP_actin_to_total_actin()
        # daughter_ratio = analyzer.analyze_ratio_of_daughter_filament_actin
        # _to_total_filamentous_actin()
        # mother_lengths = analyzer.analyze_mother_filament_lengths()
        # daughter_lengths = analyzer.analyze_daughter_filament_lengths()
        # bound_arp_ratio = analyzer.analyze_ratio_of_bound_to_total_arp23()
        # capped_ratio = analyzer.analyze_ratio_of_capped_ends_to_total_ends()
        # branch_angles = analyzer.analyze_branch_angles()
        # short_helix_pitches = analyzer.analyze_short_helix_pitches()
        # long_helix_pitches = analyzer.analyze_long_helix_pitches()
        # straightness = analyzer.analyze_filament_straightness()
        # reactions = analyzer.analyze_all_reaction_events_over_time()
        # free_actin = analyzer.analyze_free_actin_concentration_over_time()

    @staticmethod
    def visualize_actin(path_to_readdy_h5, box_size, plots):
        """
        visualize an actin trajectory in Simularium
        """
        # radii
        extra_radius = 1.5
        actin_radius = 2.0 + extra_radius
        arp23_radius = 2.0 + extra_radius
        cap_radius = 3.0 + extra_radius
        radii = {
            "arp2": arp23_radius,
            "arp2#branched": arp23_radius,
            "arp3": arp23_radius,
            "arp3#ATP": arp23_radius,
            "arp3#new": arp23_radius,
            "arp3#new_ATP": arp23_radius,
            "cap": cap_radius,
            "cap#new": cap_radius,
            "cap#bound": cap_radius,
            "actin#free": actin_radius,
            "actin#free_ATP": actin_radius,
            "actin#new": actin_radius,
            "actin#new_ATP": actin_radius,
            "actin#1": actin_radius,
            "actin#2": actin_radius,
            "actin#3": actin_radius,
            "actin#ATP_1": actin_radius,
            "actin#ATP_2": actin_radius,
            "actin#ATP_3": actin_radius,
            "actin#pointed_1": actin_radius,
            "actin#pointed_2": actin_radius,
            "actin#pointed_3": actin_radius,
            "actin#pointed_ATP_1": actin_radius,
            "actin#pointed_ATP_2": actin_radius,
            "actin#pointed_ATP_3": actin_radius,
            "actin#barbed_1": actin_radius,
            "actin#barbed_2": actin_radius,
            "actin#barbed_3": actin_radius,
            "actin#barbed_ATP_1": actin_radius,
            "actin#barbed_ATP_2": actin_radius,
            "actin#barbed_ATP_3": actin_radius,
            "actin#branch_1": actin_radius,
            "actin#branch_ATP_1": actin_radius,
            "actin#branch_barbed_1": actin_radius,
            "actin#branch_barbed_ATP_1": actin_radius,
        }
        # type grouping
        type_grouping = {
            "arp2": ["arp2", "arp2#new"],
            "arp2#ATP": ["arp2#ATP", "arp2#new_ATP"],
            "cap": ["cap", "cap#new"],
            "actin#free": ["actin#free", "actin#new"],
            "actin#free_ATP": ["actin#free_ATP", "actin#new_ATP"],
            "actin": ["actin#1", "actin#2", "actin#3"],
            "actin#ATP": ["actin#ATP_1", "actin#ATP_2", "actin#ATP_3"],
            "actin#pointed": ["actin#pointed_1", "actin#pointed_2", "actin#pointed_3"],
            "actin#pointed_ATP": [
                "actin#pointed_ATP_1",
                "actin#pointed_ATP_2",
                "actin#pointed_ATP_3",
            ],
            "actin#barbed": ["actin#barbed_1", "actin#barbed_2", "actin#barbed_3"],
            "actin#barbed_ATP": [
                "actin#barbed_ATP_1",
                "actin#barbed_ATP_2",
                "actin#barbed_ATP_3",
            ],
            "actin#branch": ["actin#branch_1"],
            "actin#branch_ATP": ["actin#branch_ATP_1"],
            "actin#branch_barbed": ["actin#branch_barbed_1"],
            "actin#branch_barbed_ATP": ["actin#branch_barbed_ATP_1"],
        }
        # convert
        data = ReaddyData(
            meta_data=MetaData(
                box_size=np.array([box_size, box_size, box_size]),
            ),
            timestep=0.1,
            path_to_readdy_h5=path_to_readdy_h5,
            radii=radii,
            type_grouping=type_grouping,
            time_units=UnitData("ns"),
            spatial_units=UnitData("nm"),
            plots=plots,
        )
        converter = ReaddyConverter(data)
        converter.write_JSON(path_to_readdy_h5)
