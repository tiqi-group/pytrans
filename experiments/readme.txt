Please write a short description of each new experimental file you create here, and which functions it exposes for use in other experiments. -VN

loading_conveyor.py : implements functions for:
		    static_waveform - function to generate a single well
		    transport_waveform - function to generate a single well whose parameters change from start to end
		    transport_waveform_multiple - as above, solving for multiple simultaneous wells
		    conveyor_waveform - loading waveform that merges together two independent wells, then recreates the well originally in the loading zone
		    reordering_waveform - dual-species waveform that creates a DC push, then a DC twist, then undoes the push, then undoes the twist. Designed to reorder a crystal into a deterministic arrangement.
		    loading_conveyor - 

trans_single.py : various routines for transport and shallow wells, used to measure the trap heating rate (20.09.2016)

load_split_swept.py: splitting waveform file with many waveforms with which parameters can be swept by scanning in Ionizer
