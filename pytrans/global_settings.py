# Different convex optimisation solvers
# global_solver = 'CVXOPT'
# global_solver = 'ECOS'
# global_solver = 'MOSEK'
# global_solver = 'GUROBI'
# global_solver = 'OSQP'
# global_solver = 'SCS'

# Solver verbosity (True/False)
global_settings = {
    'solver': 'MOSEK',
    'solver_verbose': True,
    'solver_print_weights': False,
    'solver_print_end_problem': False,
    'solver_check_end_constraints_met': False
}

global_des_pot_settings = {
    'energy_threshold': 0.004,  # in eV
    'roi_well': 0,  # which well, if there is more than 1, to use for estimating a ROI for the solver
    'roi_timestep': 0,
}

solver_weights = {
    # Cost function parameters
    'r0': 1e-15,  # punishes deviations from r0_u_ss. Can be used to set default voltages for less relevant electrodes.
    'r1': 1e-5,  # punishes the first time derivative of u, thus reducing the slew rate
    'r2': 0,  # punishes the second time derivative of u, thus further enforcing smoothness

    # default voltage for the electrodes. any deviations from
    # this will be punished, weighted by r0 and r0_u_weights
    'r0_u_ss': np.ones(num_electrodes) * default_elec_voltage,  # default voltages for the electrodes
    'r0_u_weights': np.ones(num_electrodes)  # use this to put different weights on outer electrodes
}

global_exp_pos = 20  # in um

# Global debugging
global_verbosity = 2


def print_debug(*args, **kwargs):
    if global_verbosity > 2:
        print(*args, **kwargs)
