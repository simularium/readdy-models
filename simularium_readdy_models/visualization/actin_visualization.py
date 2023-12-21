#!/usr/bin/env python

from typing import Any, Dict, List

import numpy as np
from simulariumio import (
    DISPLAY_TYPE,
    CameraData,
    DisplayData,
    MetaData,
    ScatterPlotData,
    HistogramPlotData,
    TrajectoryData,
    UnitData,
    BinaryWriter,
)
from simulariumio.filters import MultiplyTimeFilter
from simulariumio.readdy import ReaddyConverter, ReaddyData
from simulariumio.plot_readers import ScatterPlotReader, HistogramPlotReader
from subcell_analysis import SpatialAnnotator, CompressionAnalyzer
from subcell_analysis.readdy import ReaddyPostProcessor, ReaddyLoader, FrameData

from ..actin import ActinAnalyzer, ActinStructure


class ActinVisualization:
    """
    Actin-specific visualization functions to create plots,
    annotate spatial data, and save simularium files.
    """

    TIMESTEP: float = 0.1

    @staticmethod
    def ACTIN_DISPLAY_DATA(longitudinal_bonds: bool) -> Dict[str, DisplayData]:
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
    def get_bond_stretch_plot(
        trajectory: List[FrameData], 
        box_size: np.ndarray, 
        periodic_boundary: bool, 
        stride: int, 
        times: np.ndarray,
    ) -> ScatterPlotData:
        """
        Add a scatter plot of difference in bond length from ideal
        for lateral and longitudinal actin bonds vs time.
        """
        (stretch_lat, stretch_long,) = ActinAnalyzer.analyze_bond_stretch(
            trajectory, box_size, periodic_boundary, stride
        )
        mean_lat = np.mean(stretch_lat, axis=1)
        # stddev_lat = np.std(stretch_lat, axis=1)
        mean_long = np.mean(stretch_long, axis=1)
        # stddev_long = np.std(stretch_long, axis=1)
        return ScatterPlotData(
            title="Bond stretch",
            xaxis_title="T (ms)",
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
        trajectory: List[FrameData], 
        box_size: np.ndarray, 
        periodic_boundary: bool, 
        stride: int, 
        times: np.ndarray,
    ) -> ScatterPlotData:
        """
        Add a scatter plot of difference in angles from ideal
        for lateral and longitudinal actin bonds vs time.
        """
        (
            stretch_lat_lat,
            stretch_lat_long,
            stretch_long_long,
        ) = ActinAnalyzer.analyze_angle_stretch(
            trajectory, box_size, periodic_boundary, stride
        )
        mean_lat_lat = np.mean(stretch_lat_lat, axis=1)
        mean_lat_long = np.mean(stretch_lat_long, axis=1)
        mean_long_long = np.mean(stretch_long_long, axis=1)
        return ScatterPlotData(
            title="Angle stretch",
            xaxis_title="T (ms)",
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
        trajectory: List[FrameData], 
        box_size: np.ndarray, 
        periodic_boundary: bool, 
        stride: int, 
        times: np.ndarray,
    ) -> ScatterPlotData:
        """
        Add a scatter plot of difference in angles from ideal
        for lateral and longitudinal actin bonds vs time.
        """
        (
            stretch_lat_lat_lat,
            stretch_long_long_long,
        ) = ActinAnalyzer.analyze_dihedral_stretch(
            trajectory, box_size, periodic_boundary, stride
        )
        mean_lat_lat_lat = np.mean(stretch_lat_lat_lat, axis=1)
        mean_long_long_long = np.mean(stretch_long_long_long, axis=1)
        return ScatterPlotData(
            title="Dihedral stretch",
            xaxis_title="T (ms)",
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
    def generate_actin_structure_plots(
        trajectory: List[FrameData],
        box_size: np.ndarray,
        periodic_boundary: bool = True,
        plots: Dict[str, List[ScatterPlotData or HistogramPlotData]] = None,
    ) -> Dict[str, List[ScatterPlotData or HistogramPlotData]]:
        """
        Use an ActinAnalyzer to generate plots of observables
        for a diffusing actin filament.
        """
        STRIDE = 10
        if plots is None:
            plots = {
                "scatter": [],
                "histogram": [],
            }
        times = np.array([frame.time for frame in trajectory])
        plots["scatter"] += [
            ActinVisualization.get_bond_stretch_plot(
                trajectory, box_size, periodic_boundary, STRIDE, times
            ),
            ActinVisualization.get_angle_stretch_plot(
                trajectory, box_size, periodic_boundary, STRIDE, times
            ),
            ActinVisualization.get_dihedral_stretch_plot(
                trajectory, box_size, periodic_boundary, STRIDE, times
            ),
        ]
        return plots

    @staticmethod
    def generate_actin_compression_plots(
        axis_positions: List[List[np.ndarray]],
        timestep: float,
        plots: Dict[str, List[ScatterPlotData or HistogramPlotData]] = None,
    ) -> Dict[str, List[ScatterPlotData or HistogramPlotData]]:
        """
        Use an ActinAnalyzer to generate plots of observables
        for actin being compressed.
        """
        if plots is None:
            plots = {
                "scatter": [],
                "histogram": [],
            }
        perp_dist = []
        bending_energy = []
        non_coplanarity = []
        peak_asym = []
        contour_length = []
        total_steps = len(axis_positions)
        control_pts = ReaddyPostProcessor.linear_fiber_control_points(axis_positions, 10.)
        for time_ix in range(total_steps):
            perp_dist.append(CompressionAnalyzer.get_average_distance_from_end_to_end_axis(
                polymer_trace=control_pts[time_ix][0],
            ))
            bending_energy.append(1000. * CompressionAnalyzer.get_bending_energy_from_trace(
                polymer_trace=control_pts[time_ix][0],
            ))
            non_coplanarity.append(CompressionAnalyzer.get_third_component_variance(
                polymer_trace=control_pts[time_ix][0],
            ))
            peak_asym.append(CompressionAnalyzer.get_asymmetry_of_peak(
                polymer_trace=control_pts[time_ix][0],
            ))
            contour_length.append(CompressionAnalyzer.get_contour_length_from_trace(
                polymer_trace=control_pts[time_ix][0],
            ))
        times = timestep * np.arange(total_steps)
        plots["scatter"] += [
            ScatterPlotData(
                title="Average Perpendicular Distance",
                xaxis_title="T (ms)",
                yaxis_title="distance (nm)",
                xtrace=times,
                ytraces={
                    "<<<": np.zeros(times.shape),
                    ">>>": 85.0 * np.ones(times.shape),
                    "filament" : np.array(perp_dist),
                },
                render_mode="lines"
            ),
            ScatterPlotData(
                title="Bending Energy",
                xaxis_title="T (ms)",
                yaxis_title="energy",
                xtrace=times,
                ytraces={
                    "<<<": np.zeros(times.shape),
                    ">>>": 10.0 * np.ones(times.shape),
                    "filament" : np.array(bending_energy),
                },
                render_mode="lines"
            ),
            ScatterPlotData(
                title="Non-coplanarity",
                xaxis_title="T (ms)",
                yaxis_title="3rd component variance from PCA",
                xtrace=times,
                ytraces={
                    "<<<": np.zeros(times.shape),
                    ">>>": 0.03 * np.ones(times.shape),
                    "filament" : np.array(non_coplanarity),
                },
                render_mode="lines"
            ),
            ScatterPlotData(
                title="Peak Asymmetry",
                xaxis_title="T (ms)",
                yaxis_title="normalized peak distance",
                xtrace=times,
                ytraces={
                    "<<<": np.zeros(times.shape),
                    ">>>": 0.5 * np.ones(times.shape),
                    "filament" : np.array(peak_asym),
                },
                render_mode="lines"
            ),
            ScatterPlotData(
                title="Contour Length",
                xaxis_title="T (ms)",
                yaxis_title="filament contour length (nm)",
                xtrace=times,
                ytraces={
                    "<<<": 480 * np.ones(times.shape),
                    ">>>": 505 * np.ones(times.shape),
                    "filament" : np.array(contour_length),
                },
                render_mode="lines"
            ),
        ]
        return plots
    
    @staticmethod
    def add_plots_to_trajectory(
        trajectory: TrajectoryData, 
        plots: Dict[str,List[ScatterPlotData or HistogramPlotData]]
    ) -> TrajectoryData:
        reader = ScatterPlotReader()
        for plot in plots["scatter"]:
            trajectory.plots.append(reader.read(plot))
        reader = HistogramPlotReader()
        for plot in plots["histogram"]:
            trajectory.plots.append(reader.read(plot))
        return trajectory

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
        for actin being compressed.
        """
        if visualize_edges:
            edges = post_processor.edge_positions()
            traj_data = SpatialAnnotator.add_fiber_agents(
                traj_data,
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
                normal_length=10.0,
            )
            traj_data = SpatialAnnotator.add_fiber_agents(
                traj_data,
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
                segment_length=10.0,
            )
            sphere_positions = []
            for time_ix in range(len(control_points)):
                sphere_positions.append(control_points[time_ix][0])
            traj_data = SpatialAnnotator.add_sphere_agents(
                traj_data,
                sphere_positions,
                type_name="fiber point",
                radius=0.8,
                color="#eaeaea",
            )
        return traj_data

    @staticmethod
    def simularium_trajectory(
        path_to_readdy_h5: str,
        box_size: np.ndarray,
        total_steps: int,
        saved_frames: float,
        longitudinal_bonds: bool = True,
    ) -> TrajectoryData:
        """
        Get a TrajectoryData to visualize an actin trajectory in Simularium.
        """
        data = ReaddyData(
            timestep=1e-6 * (ActinVisualization.TIMESTEP * total_steps / saved_frames),
            path_to_readdy_h5=path_to_readdy_h5,
            meta_data=MetaData(
                box_size=box_size,
                camera_defaults=CameraData(
                    position=np.array([0.0, 0.0, 300.0]),
                    look_at_position=np.zeros(3),
                    up_vector=np.array([0.0, 1.0, 0.0]),
                    fov_degrees=120.0,
                ),
                scale_factor=1.0,
            ),
            display_data=ActinVisualization.ACTIN_DISPLAY_DATA(longitudinal_bonds),
            time_units=UnitData("ms"),
            spatial_units=UnitData("nm"),
        )
        return ReaddyConverter(data)._data
    
    @staticmethod
    def analyze_and_visualize_trajectory(
        output_name: str,
        total_steps: float,
        parameters: Dict[str, Any], 
        save_pickle: bool = False
    ) -> TrajectoryData:
        """
        Do all visualization, annotation, and metric calculation on an actin trajectory
        and save it as a simularium file.
        """
        # get analysis parameters
        plot_actin_structure = parameters.get("plot_actin_structure", False) 
        plot_actin_compression = parameters.get("plot_actin_compression", False) 
        visualize_edges = parameters.get("visualize_edges", False) 
        visualize_normals = parameters.get("visualize_normals", False) 
        visualize_control_pts = parameters.get("visualize_control_pts", False)
        
        # convert to simularium
        traj_data = ActinVisualization.simularium_trajectory(
            path_to_readdy_h5=output_name + ".h5",
            box_size=parameters["box_size"],
            total_steps=total_steps,
            saved_frames=1e3,
            longitudinal_bonds=bool(parameters.get("longitudinal_bonds", True)),
        )
        
        # load different views of ReaDDy data
        post_processor = None
        fiber_chain_ids = None
        axis_positions = None
        new_chain_ids = None
        time_step = max(1, total_steps * 1E-3) * parameters.get("internal_timestep", 0.1) * 1E-6  # ms
        if visualize_normals or visualize_control_pts or visualize_edges or plot_actin_structure or plot_actin_compression: 
            periodic_boundary = parameters.get("periodic_boundary", False) 
            post_processor = ReaddyPostProcessor(
                trajectory=ReaddyLoader(
                    h5_file_path=output_name + ".h5",
                    min_time_ix=0,
                    max_time_ix=-1,
                    time_inc=1,
                    timestep=time_step,
                    save_pickle_file=save_pickle,
                ).trajectory(),
                box_size=parameters["box_size"],
                periodic_boundary=periodic_boundary,
            )
            if visualize_normals or visualize_control_pts or plot_actin_compression:
                fiber_chain_ids = post_processor.linear_fiber_chain_ids(
                    start_particle_phrases=["pointed"],
                    other_particle_types=[
                        "actin#",
                        "actin#ATP_",
                        "actin#mid_",
                        "actin#mid_ATP_",
                        "actin#fixed_",
                        "actin#fixed_ATP_",
                        "actin#mid_fixed_",
                        "actin#mid_fixed_ATP_",
                        "actin#barbed_",
                        "actin#barbed_ATP_",
                        "actin#fixed_barbed_",
                        "actin#fixed_barbed_ATP_",
                    ],
                    polymer_number_range=5,
                )
                axis_positions, new_chain_ids = post_processor.linear_fiber_axis_positions(
                    fiber_chain_ids=fiber_chain_ids,
                    ideal_positions=ActinStructure.mother_positions[2:5],
                    ideal_vector_to_axis=ActinStructure.vector_to_axis(),
                )
        
        # create plots
        plots = None
        if plot_actin_compression:
            print("plot actin compression metrics")
            plots = ActinVisualization.generate_actin_compression_plots(
                axis_positions,
                time_step,
                plots=plots,
            )
        if plot_actin_structure:
            print("plot actin structure metrics")
            plots = ActinVisualization.generate_actin_structure_plots(
                post_processor.trajectory,
                parameters["box_size"],
                periodic_boundary=True,
                plots=plots,
            )
        traj_data = ActinVisualization.add_plots_to_trajectory(traj_data, plots)

        # add annotation objects to the spatial data
        traj_data = ActinVisualization.add_spatial_annotations(
            traj_data,
            post_processor,
            visualize_edges,
            visualize_normals,
            visualize_control_pts,
            new_chain_ids,
            axis_positions,
        )
        
        # save simularium file
        BinaryWriter.save(
            trajectory_data=traj_data,
            output_path=output_name + ".h5",
            validate_ids=False,
        )
        return traj_data
