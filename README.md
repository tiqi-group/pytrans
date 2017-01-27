## Pytrans, a library for creating and modifying DEATH transport waveforms

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
