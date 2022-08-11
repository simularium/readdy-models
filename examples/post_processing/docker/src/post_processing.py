#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Re-process visualization of ReaDDy model"
    )
    parser.add_argument(
        "file_name", help="the name of a file in the local directory to process"
    )
    args = parser.parse_args()
    print(f"Re-processing {args.file_name}")
    directory = os.getcwd()
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(path):
            print(path)


if __name__ == "__main__":
    main()
