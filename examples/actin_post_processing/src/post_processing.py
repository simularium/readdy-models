#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import boto3
import argparse

from visualize_actin import visualize_actin


S3_BUCKET_NAME = "readdy-working-bucket"

EXPERIMENT_NAME = "actin_twist_bend_long_bonds"

REPLICATES = 3

EXPERIMENT_CONDITIONS = [
    "tangent_fixed",
    "tangent_free",
    "radial_fixed",
    "radial_free",
    "long_tangent_fixed",
    "long_tangent_free",
    "long_radial_fixed",
    "long_radial_free",
]

ACTIN_NUMBER_TYPES = 4 * [3] + 4 * [5]


def main():
    parser = argparse.ArgumentParser(
        description="Parses an actin hdf5 (*.h5) trajectory file produced\
         by the ReaDDy software and converts it into the Simularium\
         visualization-data-format with plots"
    )
    parser.add_argument(
        "condition_index", help="index for condition to post-process"
    )
    args = parser.parse_args()
    s3 = boto3.client('s3')
    condition_index = int(args.condition_index)
    condition = EXPERIMENT_CONDITIONS[condition_index]
    print(f"Post-processing {condition}")
    # download
    if not os.path.exists("h5_files/"):
        print("Downloading from S3...")
        os.mkdir("h5_files/")
        for rep in range(REPLICATES):
            file_name = f"{EXPERIMENT_NAME}_{condition}_{rep}"
            s3.download_file(S3_BUCKET_NAME, f"outputs/{file_name}.h5", f"h5_files/{file_name}.h5")
            # s3.download_file(S3_BUCKET_NAME, f"outputs/{file_name}.dat", f"h5_files/{file_name}.dat")
    # visualize
    visualize_actin(
        actin_number_types=ACTIN_NUMBER_TYPES[condition_index],
        dir_path="h5_files/",
        box_size=600,
        total_steps=1e5,
        experiment_name=f"{EXPERIMENT_NAME}_",
        condition_name=condition,
        periodic_boundary=False,
        plot_bend_twist=True,
        plot_polymerization=False,
        save_in_one_file=True,
        color_by_run=True,
        visualize_edges=True,
        visualize_normals=True,
    )


if __name__ == "__main__":
    main()
