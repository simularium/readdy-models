!!! **This repository is being actively developed and is not yet released, proceed with caution :)**

---

## Simularium repositories
This repository is part of the Simularium project ([simularium.allencell.org](https://simularium.allencell.org)), which includes repositories:
- [simulariumIO](https://github.com/simularium/simulariumio) - Python package that converts simulation outputs to the format consumed by the Simularium viewer website
- [simularium-engine](https://github.com/simularium/simularium-engine) - C++ backend application that interfaces with biological simulation engines and serves simulation data to the front end website
- [simularium-viewer](https://github.com/simularium/simularium-viewer) - NPM package to view Simularium trajectories in 3D
- [simularium-website](https://github.com/simularium/simularium-website) - Front end website for the Simularium project, includes the Simularium viewer

# Simularium Models Util

[![Build Status](https://github.com/simularium/simularium_readdy_models/workflows/Build%20Main/badge.svg)](https://github.com/simularium/simularium_readdy_models/actions)
[![Documentation](https://github.com/simularium/simularium_readdy_models/workflows/Documentation/badge.svg)](https://simularium.github.io/simularium_readdy_models/)
[![Code Coverage](https://codecov.io/gh/simularium/simularium_readdy_models/branch/main/graph/badge.svg)](https://codecov.io/gh/simularium/simularium_readdy_models)

Tools for building computational biology models and example models from the Simularium project.

Currently includes coarse-grained monomer ReaDDy models for actin, microtubules, and kinesin.

---

## Quickstart

See [examples/README.md](examples/README.md) to run example actin, microtubules, or kinesin models either locally or on AWS with Docker.

## Installation

**Stable Release:** `pip install simularium_readdy_models`<br>
**Development Head:** `pip install git+https://github.com/simularium/simularium_readdy_models.git`<br>
**Local Editable Install** `pip install -e .[dev]` (or `pip install -e .\[dev\]` on mac) from repo root directory


## Documentation

For full package documentation please visit [simularium.github.io/simularium_readdy_models](https://simularium.github.io/simularium_readdy_models).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.


**Allen Institute Software License**

