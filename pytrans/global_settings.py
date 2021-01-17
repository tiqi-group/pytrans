# Different convex optimisation solvers
# global_solver = 'CVXOPT'
# global_solver = 'ECOS'
# global_solver = 'MOSEK'
# global_solver = 'GUROBI'
# global_solver = 'OSQP'
# global_solver = 'SCS'

# Solver verbosity (True/False)
global_settings = {
    'solver':'MOSEK',
    'solver_verbose':True,
    'solver_print_weights':False,
    'solver_print_end_problem':False,
    'solver_check_end_constraints_met':False
    }

global_des_pot_settings = {
    'energy_threshold': 0.004, # in eV
    'roi_well': 0, # which well, if there is more than 1, to use for estimating a ROI for the solver
    'roi_timestep': 0,
    }

global_exp_pos = 20 # in um

# Global debugging
global_verbosity = 2

def print_debug(*args, **kwargs):
    if global_verbosity > 2:
        print(*args, **kwargs)
