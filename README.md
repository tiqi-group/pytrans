# Pytrans, a library for creating and modifying DEATH transport waveforms

## Who should use pytrans and when?
The whole point of pytrans is to generate fast and dynamic waveforms for transport/splitting to be played back by the DEATH (our home-built fast DAC/AWG). If you want to do that, then go ahead and use pytrans!
However, if you just want to generate and analyze a static well, then it is probably a lot easier to just do that stand-alone rather than using pytrans and all its layers of abstraction and arcane file handling.
Historically, pytrans has been developed for the segtrap (i.e. a deep trap with nice symmetry and separate shim electrodes). Robin has adapted it to be somewhat useable with a surface trap (Sandia HoA2), but it is far from polished at the moment.

## Dependencies
pytrans is built on top of the default python libraries for numerical calculations and plotting (scipy, numpy, matplotlib). Additionally, it uses 'cvxpy' in order to facilitate formulating the optimization problem. You can install it and its dependencies using 'pip install cvxpy'.

## Installation files
In addition to the files present in this git repository, you also need the following files:
- global_settings.py
This file specifies which solver to use. By default it uses 'ECOS', which should be present if you have installed the dependencies correctly. For simple waveform calculations it should be enough. For more complicated problems it makes sense to use better solvers, i.e. Gurobi or MOSEK. Talk to Robin or Vlad regarding setting them up.
You can get it from J:\Temp\Robin\pytrans\global_settings.py.
Put it into the main pytrans folder, i.e. \pytrans\global_settings.py
- trap.pickle
This file contains the data from the BEM simulation for Daniels 3d trap packaged in a way that is easy and fast to import into python.
You can get it from J:\Temp\Robin\pytrans\moments_file\trap.pickle
Put it into \pytrans\moments_file\

## Usage

Please write a short description of each new file you create here, and which functions it exposes for use in experiments.

__loading_utils.py__ : implements the basic loading waveform and saves it to a waveform file. All the other experiments rely on it to generate the basic loading and reordering parts of their waveform file.

__transport_utils.py__ : implements functions for:
- static_waveform - generate a single waveform with position, frequency (assuming 40Ca+ ion) and DC offset as the main inputs
- transport_waveform - generate a single waveform whose parameters are swept in time
- transport_waveform_multiple - like transport_waveform, solving for multiple simultaneous wells
- conveyor_waveform - generate a waveform that merges together two independent wells, then recreates the well originally in the loading zone (used for loading only)
- reordering_waveform - mixed-species waveform that creates a DC push, then a DC twist, then undoes the push, then undoes the twist. Designed to reorder a mixed-species crystal into a deterministic arrangement of ions

__trans_single.py__ : various routines for transport and shallow wells, used to measure the trap heating rate (20.09.2016)

__load_split_swept.py__ : splitting waveform file with many waveforms with which parameters can be swept by scanning in Ionizer
