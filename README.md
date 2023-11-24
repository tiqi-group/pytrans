# pytrans: potentials and waveforms for trapped ions <!-- omit from toc -->

[![License](https://img.shields.io/badge/License-AGPLv3-firebrick.svg?style=flat-square)](https://opensource.org/license/agpl-v3/)
[![Documentation Status](https://readthedocs.org/projects/pytrans/badge/?version=latest)](https://pytrans.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/722720016.svg)](https://zenodo.org/doi/10.5281/zenodo.10204441)

Pytrans is a python package for creating static and dynamic potentials for ion traps. Starting from a description of the trap geometry and its electrical properties, it allows finding optimal sets of voltages (often called *waveforms* if time-dependent) producing the target trapping potential. It allows to evaluate the equilibrium configuration of an ensamble of ions trapped in the resulting potential and to simulate their classical dynamics.

While the solution depends on the specifics of the trap for which it has been generated, pytrans is trap-agnostic: it can model different types of ion traps, wrapping specific data into generic structures that implement waveform generation and analysis in a unified way.

It uses [cvxpy](https://www.cvxpy.org/index.html) as optimization backend, and numpy, scipy and matplotlib for analysis, simulation, and visualization. Pytrans is open source and tested for Python >= 3.10 on all Unix and Windows platforms.

- [Installation](#installation)
  - [pip](#pip)
  - [poetry](#poetry)
  - [Install from source](#install-from-source)
- [Documentation](#documentation)
- [Usage](#usage)
  - [Trapping Ca40 in a Paul trap](#trapping-ca40-in-a-paul-trap)
- [Contributing](#contributing)
- [License](#license)
- [Authors and history](#authors-and-history)
- [References](#references)

## Installation

### pip

Pytrans is available on PyPI. You can install it via `pip` by

```bash
pip install pytrans
```

Additionally, to use other solvers suported by cvxpy, it is necessary to install additional packages. Follow the installation instructions on <https://www.cvxpy.org/install/>.

### poetry

When using pytrans as part of a project, including e.g. one or more trap models and other specific tools, we recommend using a project manager tool like `poetry`. The [pytrans-examples](https://github.com/tiqi-group/pytrans-examples) repo is a reference implementation of a poetry project dedicated to generating waveforms for specific ion traps.

### Install from source

Clone a fresh copy of the source repository, perhaps within a virtual environment, and install it in editable mode including both the required and the optional dependencies for code linting and documentation.

```bash
git clone https://github.com/tiqi-group/pytrans
cd pytrans
pip install -e .[dev,docs]
```

Pytrans requires python >= 3.10.

## Documentation

The latest documentation can be found on <https://pytrans.readthedocs.io>.

In [pytrans-examples](https://github.com/tiqi-group/pytrans-examples/tree/main/examples) we provide numerous examples of waveform generation and potential analysis in two different types of ion traps.

## Usage

A typical usage of pytrans involves:

- implementing a model for a specific ion trap
- defining a target potential and setting up an optimization problem to reproduce it in the trap
- analyzing and visualizing the results
  
### Trapping Ca40 in a Paul trap

Here we find a set of voltages suitable for traping one Ca40 ion in a 3D, segmented, microfabricated Paul trap, in a potential well with an axial oscillation frequency of 1 MHz. Extract from the example notebook [01_static_potential.ipynb](https://github.com/tiqi-group/pytrans-examples/tree/main/examples/01_static_potential.ipynb).

```python
trap = SegmentedTrap()

n_samples = 1
waveform = init_waveform(n_samples, trap.n_electrodes)

r0 = (0, 0, trap.z0)
axial_curv = freq_to_curv(1e6, ion=Ca40)

objectives = [
    obj.GradientObjective(waveform[0], trap, *r0, value=0, ion=Ca40),
    obj.HessianObjective(waveform[0], trap, *r0, entries='xx', value=axial_curv, ion=Ca40),

    obj.VoltageObjective(waveform, 10, constraint_type='<='),
    obj.VoltageObjective(waveform, -10, constraint_type='>='),
]

res = solver(objectives, verbose=True)
waveform = res.waveform.value  # optimal value, np array
```

## Contributing

All contributions are welcome! Use the [issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues) to report bugs or feature requests, or add your own code by forking the project and [opening a pull request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

## License

Pytrans is open source and released under the GNU Affero General Public License version 3 [(AGPLv3)](https://opensource.org/license/agpl-v3/).

## Authors and history

The project has been developed in the [Trapped Ion Quantum Information](https://tiqi.ethz.ch/) (TIQI) group at [ETH Zurich](https://ethz.ch/). Started as a collection of python scripts for generating shuttling waveforms in a specific ion trap, it received contributions from numerous members of the group, ranging from students to PhDs and postdocs. It is currently used to model all the ion traps used in TIQI projects.

Contributors, in a non-strictly-cronological order:

- Vlad Negnevitsky
- Matteo Marinelli
- Francesco Lancellotti
- Robin Ostwald
- Sebastian Heinekamp
- Tobias Sagesser
- Carmelo Mordini
- Yuto Motohashi
- Michalis Theodorou

## References

Articles or theses describing the projects through which this project has been developed, and further references.

- V. Negnevitsky, Feedback-stabilised quantum states in a mixed-species ion system, [PhD thesis](https://www.research-collection.ethz.ch/handle/20.500.11850/295923)
- D. Leibfried et al., Quantum dynamics of single trapped ions, <https://doi.org/10.1103/RevModPhys.75.281>
- J. P. Home, Quantum science and metrology with mixed-species ion chains, <https://arxiv.org/abs/1306.5950>

The logic of the modular cost function was inspired by the [electrode](https://github.com/nist-ionstorage/electrode) package by Robert Jordens.

About documentation:

- <https://diataxis.fr/>
- <https://github.com/mhucka/readmine>
