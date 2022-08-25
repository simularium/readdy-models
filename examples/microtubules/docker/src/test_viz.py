from simularium_models_util.visualization import (
    MicrotubulesVisualization,
    microtubules_visualization,
)
from simularium_models_util.microtubules import MICROTUBULES_REACTIONS
from simularium_models_util import ReaddyUtil
from simulariumio import FileConverter, InputFileData
import numpy as np

# h5_path = "outputs/scan_1_conc1x_rxn1x.h5" # sim_steps = 1e3
# h5_path = "outputs/test_conc1x_rxn1x.h5"
# h5_path = "outputs/microtubules_scan_growth_attach_20220418_x1_growth_x1_attach_1.h5" # sim_steps = 2e7
# h5_path = "outputs/test_attach_rxn_run0.h5"
# h5_path = "outputs/test_attach_rxn_run1.h5"
# h5_path = "outputs/test_disable_ignore_run1.h5"
h5_path = "outputs/debug_attach_rxn_run1.h5"
# h5_path = "outputs/single_tubulin_site_remove_zero_length_MTs.h5"

# box_size = np.array([150.0, 150.0, 250.0])
box_size = np.array([300.0] * 3)
stride = 1
pickle_file_path = (
    "/mnt/c/Users/saurabh.mogre/OneDrive - Allen Institute/Projects/AICS/Simularium/simularium-models-util/examples/microtubules/"
    + h5_path
    + ".dat"
)
# sim_steps = 2e7
# sim_steps = 1e3
sim_steps = 1e1
viz_steps = max(int(sim_steps / 1000.0), 1)
scaled_time_step_us = 0.1 * 1e-3 * viz_steps

save_pickle_file = True

(
    monomer_data,
    reactions,
    times,
    _,
) = ReaddyUtil.monomer_data_and_reactions_from_file(
    h5_file_path=h5_path,
    stride=stride,
    timestep=0.1,
    reaction_names=MICROTUBULES_REACTIONS,
    save_pickle_file=False,
    pickle_file_path=pickle_file_path,
)

plots = MicrotubulesVisualization.generate_plots(
    monomer_data=monomer_data,
    reactions=reactions,
    times=times,
)

save_converter = True
load_from_file = True

input_path = h5_path + ".simularium"

if load_from_file:
    converter = FileConverter(InputFileData(file_path=input_path))
else:
    converter = MicrotubulesVisualization.visualize_microtubules(
        h5_path,
        box_size=box_size,
        scaled_time_step_us=scaled_time_step_us,
        plots=plots,
    )
    converter._data = ReaddyUtil._add_edge_agents(
        traj_data=converter._data,
        monomer_data=monomer_data,
        exclude_types=["tubulinA#free", "tubulinB#free"],
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
