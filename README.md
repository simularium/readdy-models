## Simularium repositories
This repository is part of the Simularium project ([simularium.allencell.org](https://simularium.allencell.org)), which includes repositories:
- [simulariumIO](https://github.com/simularium/simulariumio) - Python package that converts simulation outputs to the format consumed by the Simularium viewer website
- [octopus](https://github.com/simularium/octopus) - Python backend application that interfaces with biological simulation engines and serves simulation data to the front end website
- [simularium-viewer](https://github.com/simularium/simularium-viewer) - NPM package to view Simularium trajectories in 3D
- [simularium-website](https://github.com/simularium/simularium-website) - Front end website for the Simularium project, includes the Simularium viewer

# Simularium ReaDDy Models
Tools for building computational biology models and example models from the Simularium project. Includes coarse-grained monomer [ReaDDy](https://readdy.github.io/) models for actin, microtubules, and kinesin.

---

## Quickstart

See [examples/README.md](examples/README.md) to run example actin, microtubules, or kinesin models either locally or on AWS with Docker.

## Installation

**Stable Release:** `pip install simularium_readdy_models`<br>
**Development Head:** `pip install git+https://github.com/simularium/simularium_readdy_models.git`<br>

### Development Install with `conda`:

(`conda` is currently required to install `readdy`.)

1. Create a virtual environment with conda-specific dependencies: `conda env create -f environment.yml`
2. Activate the environment: `conda activate readdy_models`
3. Install remaining dependencies: `just install`

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

