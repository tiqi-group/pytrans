#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Created: 01-2021 - Carmelo Mordini <carmelo> <cmordini@phys.ethz.ch>
"""
Generate a 'static waveform' (so to speak) with one sample per channel

Electrode mappings are from
OneNote: Cryo-Experiment/Hardware/DC/Fastino adapter board
"""

import random
import json
import numpy as np
from colorama import Fore


N_DACS = 4 * 32
DAC_IDX = 3

MONITOR_CHANNEL = 6, "A7"

# A dictionary specifying the mapping from electrode to DAC channels
# Electrode index: DAC_p{n}
electrode_DAC_map = {1: 29,
                     2: 28,
                     3: 30,
                     4: 15,
                     5: 20,
                     6: 8,
                     7: 17,
                     8: 5,
                     9: 10,
                     10: 23,
                     11: 11,
                     12: 12,
                     13: 9,
                     14: 22,
                     15: 19,
                     16: 3,
                     17: 1,
                     18: 2,
                     19: 21,
                     20: 4,
                     21: 14,  # Mesh
                     22: 16,  # GND, FuzzButton 23
                     23: 7,  # GND, FuzzButton 37
                     24: 18,  # GND, FuzzButton 13
                     25: 27,  # GND, FuzzButton 47
                     26: MONITOR_CHANNEL[0]
                     }

DAC_electrode_map = {v: k for k, v in electrode_DAC_map.items()}

# A map from electrode index to subD25 output -- to make testing easier
# Just for printing out, not used anywhere in the actual communication
# This is for the LinoiX trap with 729 outcoupler for axial MS gate
electrode_subD_map = {1: "B8",
                      2: "B7",
                      3: "B13",
                      4: "A16",
                      5: "A23",
                      6: "A9",
                      7: "A19",
                      8: "A6",
                      9: "A11",
                      10: "B2",
                      11: "A12",
                      12: "A13",
                      13: "A10",
                      14: "B1",
                      15: "A22",
                      16: "A4",
                      17: "A2",
                      18: "A3",
                      19: "A24",
                      20: "A5",
                      21: "Mesh, A15",
                      22: "GND, A17",
                      23: "GND, A8",
                      24: "GND, A20",
                      25: "GND, B6",
                      26: f"Monitor, {MONITOR_CHANNEL[1]}"
                      }


def build_waveform_dict(index: int, samples: list, description='', generated=True, uid=None):
    if uid is None:
        uid = f'{random.randrange(2**32):08x}'
    return {f'wav{index}': {'description': description, 'generated': generated, 'samples': samples, 'uid': uid}}


def load_waveforms(file_name: str):
    with open(file_name) as f:
        d = json.load(f)
    return d


def save_waveforms(waveforms: dict, file_name: str):
    with open(file_name, 'wt') as f:
        json.dump(waveforms, f, indent='\t', sort_keys=True)


def voltages_to_wf(voltages, verbose=False):
    samples = [[0.0 for _ in range(len(voltages))] for _ in range(N_DACS)]
    # print(len(voltages))
    for sample_ix, volts in enumerate(voltages):
        for j, v in enumerate(volts):
            channel = electrode_DAC_map[j + 1] - 1
            samples[channel + DAC_IDX * 32][sample_ix] = float(v)
        # print(j + 1, channel + 1, v, samples[channel + DAC_IDX * 32])

    if verbose:
        s = -1
        print(f"Generating waveform with {len(voltages)} samples (printing {s}) {voltages.shape}")
        for j, v in enumerate(voltages[s]):
            # j, v = -1, voltages[-1][0]
            print(Fore.YELLOW + f"  Electrode {j + 1:02d} ", end='')
            print(f"({electrode_subD_map[j + 1]:s}): {v * 2.5:+.4f} V  -> [{samples[3*32 + electrode_DAC_map[j + 1] - 1][s]:+.4f}]")
            # [f"    Electrode {j + 1:02d} (DAC_p{electrode_DAC_map[j + 1]:02d}): {v * 2.5:+.4f} V  -> [{samples[3*32 + electrode_DAC_map[j + 1] - 1][0]:+.4f}]" for j, v in enumerate(voltages)]
    return samples


def wf_to_voltages(samples, verbose=False):
    n_volts = 26
    voltages = np.zeros((len(samples[0]), n_volts))
    for channel, electrode_ix in DAC_electrode_map.items():
        voltages[:, electrode_ix - 1] = samples[channel - 1 + DAC_IDX * 32]
    return voltages


def generate_waveform(voltages, index, description='', generated=True, uid=None,
                      waveform_filename=None, verbose=False):

    # print(samples[3*32:])
    samples = voltages_to_wf(voltages, verbose)
    waveforms = build_waveform_dict(index, samples, description, generated, uid)

    if waveform_filename:
        save_waveforms(waveforms, file_name=waveform_filename)
        if verbose:
            print(f"Writing to {waveform_filename}")
    return waveforms
