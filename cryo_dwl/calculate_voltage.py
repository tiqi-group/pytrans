import numpy as np
from pathlib import Path

filename = Path(__file__).resolve().parent / 'vbasis_x0.npy'
vb0 = np.load(filename)


def calculate_voltage(axial, split, tilt, x_comp, y_comp, z_comp):  # , xCubic, vMesh, vGND, xyTilt=0, xzTilt=0):
    # Array of voltages. 20 electrodes + mesh + (GND level)*4
    voltages = (axial, split, tilt, x_comp, y_comp, z_comp) @ vb0
    voltages = np.r_[voltages, np.zeros((5,))]
    return voltages
