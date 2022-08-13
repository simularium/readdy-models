#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import boto3

from visualize_actin import visualize_actin


S3_BUCKET_NAME = "readdy-working-bucket"

REPLICATES = 10

EXPERIMENT_CONDITIONS = [
    "1_radial_fixed",  # no longitudinal bonds
    "1_radial_free",
    "1_tangent_fixed",
    "1_tangent_free",
    "dihedral_strength10_radial_fixed",  # longitudinal bonds
    "dihedral_strength10_radial_free",
    "dihedral_strength10_tangent_fixed",
    "dihedral_strength10_tangent_free",
]


def get_output_name(condition):
    if "dihedral_strength" in condition:
        return "long_" + condition[20:]
    else:
        return "no-long_" + condition[2:]

def main():
    s3 = boto3.client('s3')
    for condition_index in range(len(EXPERIMENT_CONDITIONS)):
        condition = EXPERIMENT_CONDITIONS[int(condition_index)]
        output_name = get_output_name(condition)
        print(f"Re-processing {output_name}")
        # download
        if not os.path.exists("h5_files/"):
            os.mkdir("h5_files/")
            for rep in range(REPLICATES):
                input_file_name = f"outputs/actin_twist_bend_{condition}_{rep}.h5"
                output_file_name = f"actin_twist_bend_{output_name}_{rep}.h5"
                s3.download_file(S3_BUCKET_NAME, input_file_name, f"h5_files/{output_file_name}")
        # visualize
        visualize_actin(
            dir_path="h5_files/",
            box_size=200,
            total_steps=4e6,
            experiment_name=f"actin_twist_bend_{output_name}_",
            periodic_boundary=False,
            plot_bend_twist=True,
            plot_polymerization=False,
            save_in_one_file=True,
            color_by_run=True,
            visualize_edges=True,
            visualize_normals=True,
        )
        return


if __name__ == "__main__":
    main()
