## Overall test script that tests out pytrans and its functions

import sys
sys.path.append("../")
from pytrans import *

import transport_utils as tu

import unittest

class TestPytrans(unittest.TestCase):
    def test_static_solver(self):
        self.assertEqual(5, 10/2)
        self.assertAlmostEqual(134.5671000000001, 134.5671)
    # Continue here

class Test_split_waveforms_reparam(unittest.TestCase):
    def test_split_waveforms_reparam(self):
        # tu.conveyor_rec_waveform([-1870, 0],
        #                          [0.7, 1.1],
        #                          [600, 1000],
        #                          2000,
        #                          "asdf",
        #                          linspace_fn=zpspace)
    
        # Debugging only!
        load_loc = -1870
        load_freq = 0.7
        load_offs = 600

        exp_loc = 0
        exp_freq = 1.1
        exp_offs = 1000        

        rec_loc = -844
        rec_ratio = (exp_loc - rec_loc)/(exp_loc - load_loc)
        rec_freq = exp_freq + (load_freq - exp_freq)*rec_ratio
        rec_offs = exp_offs + (load_offs - exp_offs)

        split_loc = -422.5
        split_ratio = (exp_loc - split_loc)/(exp_loc - load_loc)
        split_freq = exp_freq + (load_freq - exp_freq) * split_ratio
        split_offs = exp_offs + (load_offs - exp_offs) * split_ratio

        pass
        
        well_load_to_rec = tu.transport_waveform_multiple([[load_loc, rec_loc],[exp_loc, exp_loc]],
                                                          # [[load_freq, rec_freq],[exp_freq, rec_freq]],
                                                          [[load_freq, rec_freq],[exp_freq, rec_freq]],
                                                          [[load_offs, rec_offs],[rec_offs, rec_offs]], 100, "", linspace_fn=zpspace, roi_timestep=-1)
        
        well_static = tu.transport_waveform_multiple([[rec_loc, rec_loc],[exp_loc, exp_loc]],
                                                     [[rec_freq, rec_freq],[rec_freq, rec_freq]],
                                                     [[rec_offs, rec_offs],[rec_offs, rec_offs]], 100, "", linspace_fn=zpspace)


        ax = WavPotential(well_load_to_rec).plot_one_wfm(-1)
        WavPotential(well_static).plot_one_wfm(0, ax)

        st()        
        
        # wf = split_waveforms_reparam(
        #     0, 1, 1000,
        #     [-844, 0],
        #     []

class TestExpConveyor(unittest.TestCase):
    wfm = tu.conveyor_rec_waveform(
        [-1870, 0], [0.7, 1.1], [600, 1000], 2002,
        "Load -> exp", linspace_fn=zpspace)

    wfp = WavPotential(wfm)
    st()
    
if __name__ == "__main__":
    unittest.main()
