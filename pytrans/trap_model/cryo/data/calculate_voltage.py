import numpy as np


_basis = []


def _populate_basis():
    M = np.asarray([
        [-0.26442, 0.61973, -1.9491, 0.61973],      # axial
        [0.11526, -0.28313, 0.013373, -0.28313],    # tilt
        [0, -0.025833, 0, 0.025833],                # x
        [0, 0.06717, 0.06717, 0.06717],             # y
        [0.018312, 0.11655, 0.042762, 0.11655]      # z
    ])

    sign = np.asarray([1, -1, 1, -1, 1]).reshape(-1, 1)
    M = np.concatenate([M, sign * M], axis=1)
    for zone in 1, 2, 3:
        B = np.zeros((5, 20))
        B[:, [0, 10]] = M[:, [0, 4]]
        q = 3 * zone - 2
        B[:, q:q + 3] = M[:, 1:4]
        B[:, q + 10:q + 13] = M[:, 5:8]
        _basis.append(B)


_populate_basis()


def calculate_voltage(curv, tilt, xComp, yComp, zComp, zone=2):
    assert zone in [1, 2, 3]
    x = [np.sign(curv) * curv**2, tilt, xComp, yComp, zComp]
    voltages = x @ _basis[zone - 1]
    return voltages


def vSet_axial(curv):
    # Voltage set creating 1 MHz axial frequency
    # volt_axial = [-0.25685, 0, 0, 0, 0, 0.34665, 0.34665, -1.6753, 0.34665, 0.34665,
    # -0.25685, 0, 0, 0, 0, 0.34665, 0.34665, -1.6753, 0.34665, 0.34665,
    # 0, 0, 0, 0, 0]

    # 3 electrodes, zone 2
    volt_axial = [-0.26442, 0, 0, 0, 0.61973, -1.9491, 0.61973, 0, 0, 0,
                  -0.26442, 0, 0, 0, 0.61973, -1.9491, 0.61973, 0, 0, 0,
                  0, 0, 0, 0, 0]

    # 3 electrodes, zone 3
    # volt_axial = [-0.26442, 0, 0, 0, 0, 0, 0, 0.61973, -1.9491, 0.61973,
    #               -0.26442, 0, 0, 0, 0, 0, 0, 0.61973, -1.9491, 0.61973,
    #               0, 0, 0, 0, 0]

    assert(len(volt_axial) == 25)
    return np.sign(curv) * curv**2 * np.array(volt_axial)


def vSet_tilt(tilt):
    # Voltage set creating 1MHz along y+z and -1MHz along y-z
    volt_tilt = [0.11526, 0, 0, 0, -0.28313, 0.013373, -0.28313, 0, 0, 0,
                 -0.11526, 0, 0, 0, 0.28313, -0.013373, 0.28313, 0, 0, 0,
                 0, 0, 0, 0, 0]
    assert(len(volt_tilt) == 25)
    return tilt * np.array(volt_tilt)


def vSet_xComp(xComp):
    # Voltage set displacing the ion by along x
    volt_xComp = [0, 0, 0, 0, 0, 0, 0, -0.025833, 0, 0.025833,
                  0, 0, 0, 0, 0, 0, 0, -0.025833, 0, 0.025833,
                  0, 0, 0, 0, 0]
    assert(len(volt_xComp) == 25)
    return xComp * np.array(volt_xComp)


def vSet_yComp(yComp):
    # Voltage set displacing the ion by (roughly) 1um along y
    # Assuming 3 MHz radial frequency
    volt_yComp = [0, 0, 0, 0, 0, 0, 0, 0.06717, 0.06717, 0.06717,
                  0, 0, 0, 0, 0, 0, 0, -0.06717, -0.06717, -0.06717,
                  0, 0, 0, 0, 0]
    assert(len(volt_yComp) == 25)
    return yComp * np.array(volt_yComp)


def vSet_zComp(zComp):
    # Voltage set displacing the ion by (roughly) 1um along z
    # Assuming 3 MHz radial frequency
    volt_zComp = [0.018312, 0, 0, 0, 0, 0, 0, 0.11655, 0.042762, 0.11655,
                  0.018312, 0, 0, 0, 0, 0, 0, 0.11655, 0.042762, 0.11655,
                  0, 0, 0, 0, 0]
    assert(len(volt_zComp) == 25)
    return -zComp * np.array(volt_zComp)

# dont' worry


def vSet_xCubic(xCubic):
    # Voltage set creating 1 kHz^2/um cubic potential along x
    volt_xCubic = [0, 0, 0, 0, 0, -0.000492, 0.000189, 0, -0.000189, 0.000492,
                   0, 0, 0, 0, 0, -0.000492, 0.000189, 0, -0.000189, 0.000492,
                   0, 0, 0, 0, 0]
    assert(len(volt_xCubic) == 25)
    return -xCubic * np.array(volt_xCubic)


def vSet_xyTilt(xyTilt):
    # In a 3 MHz radial well, differentially displace two ions by +- 1um along y, which are spaced by 5 um
    volt_xyTilt = [0, 0, 0, 0, 0,
                   -4.2, -4.2, 0, 4.2, 4.2,
                   0, 0, 0, 0, 0,
                   4.2, 4.2, 0, -4.2, -4.2,
                   0, 0, 0, 0, 0]
    assert(len(volt_xyTilt) == 25)
    return xyTilt * np.array(volt_xyTilt)


def vSet_xzTilt(xzTilt):
    # In a 3 MHz radial well, differentially displace two ions that are 5 um spaced by +- 1 um along z
    volt_xzTilt = [-0.20171, 0, 0, 0, 0,
                   8.2589, 8.2589, 0, -10.536, -10.536,
                   -0.1445, 0, 0, 0, 4.7086,
                   -116.51, 30.483, 0, -36.9, 127.72,
                   0, 0, 0, 0, 0]
    assert(len(volt_xzTilt) == 25)
    return xzTilt * np.array(volt_xzTilt)


def vSet_mesh(vMesh):
    # Set the mesh voltage
    volt_mesh = np.zeros(25)
    volt_mesh[20] = vMesh
    return(volt_mesh)


def vSet_gnd(vGND):
    # Set the trap GND voltage
    volt_gnd = np.zeros(25)
    # volt_gnd[21] = vGND
    # volt_gnd[22] = vGND
    # volt_gnd[23] = vGND
    # volt_gnd[24] = vGND
    return(volt_gnd)


def _calculate_voltage(curv, tilt, xComp, yComp, zComp, xCubic, vMesh, vGND, xyTilt=0, xzTilt=0):
    # Array of voltages. 20 electrodes + mesh + (GND level)*4
    voltages = np.zeros(25)

    # Sum up all contributions
    voltages = (voltages + vSet_axial(curv)
                + vSet_tilt(tilt)
                + vSet_xComp(xComp)
                + vSet_yComp(yComp)
                + vSet_zComp(zComp)
                + vSet_xCubic(xCubic)
                + vSet_mesh(vMesh)
                + vSet_gnd(vGND)
                + vSet_xyTilt(xyTilt)
                + vSet_xzTilt(xzTilt))

    return voltages


def _calculate_voltage1(curv, tilt, xComp, yComp, zComp):
    voltages = np.zeros(25)
    # Sum up all contributions
    voltages = (voltages + vSet_axial(curv)
                + vSet_tilt(tilt)
                + vSet_xComp(xComp)
                + vSet_yComp(yComp)
                + vSet_zComp(zComp)
                )

    return voltages[:20]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for zone in [1, 2, 3]:
        ax.plot(calculate_voltage(1, 0, 0, 0, 0, zone))
    plt.show()
