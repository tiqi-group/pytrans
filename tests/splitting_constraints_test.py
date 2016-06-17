#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

def splitting_constraints_test():
    # Simulate 5 electrodes, each with a very simple Gaussian envelope
    z_axis = np.linspace(-100,100,1000)*um
    e0_moment = np.exp(-(z_axis - (-60))**2 / 30**2)
    e1_moment = np.exp(-(z_axis - (-3))**2 / 30**2)
    e2_moment = np.exp(-(z_axis - (0))**2 / 30**2)
    e1_moment = np.exp(-(z_axis - (-3))**2 / 30**2)
    e2_moment = np.exp(-(z_axis - (0))**2 / 30**2)    

if __name__ == "__main__":
    splitting_constraints_test()

    z_axis = np.linspace(-100,100,1000)*um
    e0_moment = np.exp(-(z_axis - (-60*um))**2 / (45*um)**2)
    e1_moment = np.exp(-(z_axis - (-30*um))**2 / (45*um)**2)
    e2_moment = np.exp(-(z_axis - (0*um))**2 / (45*um)**2)
    e3_moment = np.exp(-(z_axis - (30*um))**2 / (45*um)**2)
    e4_moment = np.exp(-(z_axis - (60*um))**2 / (45*um)**2)        
    moments = np.column_stack([e0_moment, e1_moment, e2_moment, e3_moment, e4_moment])

    potential_quad = z_axis**2
    potential_quart = z_axis**4

    # alphas = np.array([1,0.5,0.1,-0.5,-1])*1e8
    # betas = np.array([0,1,2,3,4])*1e12
    separations = np.array([0, 2, 10, 30, 60, 90])*um
    alphas = np.array([1e12, 1e9, -1e9, -1e11, -1e12, -1e12])    
    betas = np.zeros_like(alphas)
    for k, (sep, alph) in enumerate(zip(separations[1:], alphas[1:])):
        betas[k+1] = np.abs(alph)/2/sep**2        
    print(betas)
    voffsets = np.array([0, 0, 0, 0, 0, 0])
    
    states = []
    uopt = cvy.Variable(5) # 5 electrodes
    vscale = cvy.Variable()

    alpha = cvy.Parameter()
    beta = cvy.Parameter(sign="positive")
    voffs = cvy.Parameter()

    roi_dist = 20*um
    roi_idxes = np.arange(np.argmax(z_axis>-roi_dist/2), np.argmax(z_axis>roi_dist/2))
    st()

    cost = cvy.sum_squares(moments[roi_idxes,:]*uopt
                           - (alpha*potential_quad[roi_idxes]
                              + beta*potential_quart[roi_idxes])*vscale - voffs)
    constr = [-10 <= uopt, uopt <= 10]
    # constr += [0.1 <= vscale]
    states.append(cvy.Problem(cvy.Minimize(cost-vscale), constr))
    prob = sum(states)

    for a, b, offs in zip(alphas, betas, voffsets):
        alpha.value = a
        beta.value = b
        voffs.value = offs
        prob.solve(solver='ECOS', verbose=True)
        ans = uopt.value
        for m, u in zip(moments.T, np.ravel(ans)):
            col = m.T
            plt.plot(z_axis, col*u,':')

        plt.plot(z_axis, (potential_quad*a+potential_quart*b)*vscale.value + offs, 'g')
        plt.plot(z_axis, moments*ans, 'r')

        plt.gca().set_ylim([-20, 20])
        plt.title("Alpha = "+str(a)+", beta = "+str(b)+", offset = "+str(offs))
        
        print(vscale.value)
        plt.show()
