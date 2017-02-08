#!/usr/bin/env python3
# Splitting library, building on pytrans' existing functionality
# (NOT FINISHED!)

import sys
sys.path.append("./")
from pytrans import *
import transport_utils as tu
import scipy.optimize as sopt

########## OLD CODE (used to investigate Home/Steane paper's tradeoffs) ###########

def scan_alpha_beta():
    # Look at standard
    spatial_range = 20*um
    z = np.linspace(-spatial_range/2,spatial_range/2,1000)
    a_b_traj = ((1,1e-6),(0,1),(-1,100))
    a_sc = 1e8
    b_sc = 1e18
    ramp_pts = 3

    for (a_start,b_start), (a_end, b_end) in zip(a_b_traj[:-1], a_b_traj[1:]):
        a_ramp = np.linspace(a_start, a_end, ramp_pts)*a_sc
        b_ramp = np.linspace(b_start, b_end, ramp_pts)*b_sc

        for a, b in zip(a_ramp, b_ramp):
            plt.plot(z, a*z**2+b*z**4)
            rough_d = np.sqrt(np.abs(a)/2/b)

            #st()
            
            print("rough d = ", rough_d, ", d = ", sopt.root(lambda k: dist(a, b, k), rough_d).x )

    plt.show()

def ion_distance_ab(alpha, beta, d):
    if d<0: # to constrain the solver
        return abs(d)*1e5+1e2
    return beta*d**5+2*alpha*d**3-electron_charge/2/np.pi/epsilon_0
    
def ion_distance_as(d, s, alpha):
    d2s = d/2/s
    return d2s**5 + np.sign(alpha)*d2s**3 - electron_charge/4/np.pi/epsilon_0/np.abs(alpha)/(2*s)**3

def freq_to_alpha(freq):
    return freq**2*atomic_mass_unit*40/2/electron_charge

def reproduce_fig2_home_steane():
    alpha_x = np.linspace(-1.4e8, 1.4e8, 100)
    beta = 2.7e18

    w1_y = np.zeros_like(alpha_x)
    d_y = np.zeros_like(alpha_x)

    s_x = np.sqrt(np.abs(alpha_x)/2/beta)
    for k, (s, alpha) in enumerate(zip(s_x,alpha_x)):
        d = np.abs(sopt.root(lambda m: ion_distance_as(m, s, alpha), s).x)
        d_y[k] = d

    w1_y = np.sqrt(2*alpha_x + 3*beta*d_y**2*electron_charge/atomic_mass_unit/40)

    st()

def look_at_wells_manually():
    alpha = 1.4e8
    beta = 2.65e18

    def trap_freqs(z_axis, pot):
        # pot += np.random.random(z_axis.size)*1e-10 # to avoid argrelmin getting stuck
        # Re-implementing stuff in WavPotential class
        pot_resolution=z_axis[1]-z_axis[0]
        potg2 = np.gradient(np.gradient(pot))#self.pot_resolution**2
        min_indices = ssig.argrelextrema(pot, np.less_equal, order=20)
        offsets = potg2[min_indices]
        grads = potg2[min_indices]/pot_resolution**2
        trap_freqs = np.sqrt(electron_charge*grads / (40*atomic_mass_unit))/2/np.pi
        trap_locs = z_axis[min_indices]

        return {'min_indices':min_indices, 'offsets':offsets, 'freqs':trap_freqs, 'locs':trap_locs}

    z_axis = np.linspace(-20,20,1000)*um
    single_pot = alpha*z_axis**2
    double_pot = -alpha*z_axis**2+beta*z_axis**4

    st()
    print(trap_freqs(z_axis, single_pot))
    print(trap_freqs(z_axis, double_pot))






######## NEW CODE (developed mainly in testing/splitting_constraints_test.py) #########

import scipy.interpolate as interp

def plot_split_voltages(samples, electrodes=[2,3,4,5,6,7,8]):
    plt.plot(samples[physical_electrode_transform[electrodes],:].T)
    plt.legend(list(str(k) for k in electrodes))
    plt.show()

def calc_beta(sep, alp):
    # sep: separation from ion to ion
    # alp: prefactor in quadratic
    return  2*np.abs(alp)/(sep)**2
    
def interpolate_moments(z_shifted, moments, roi_idxes, order=4):
        return np.polyfit(z_shifted[roi_idxes], moments[roi_idxes,:], order)

def generate_interp_polys(z_axis, moments, centre, roi_dist):
    # z_axis: axis corresponding to moments
    # moments: M x N matrix, M = number of z points, N = number of trap electrodes (may be reduced)
    # centre: z axis centre around which to fit polynomials (same units as z_axis)
    # roi_dist: total z range over which to fit polynomials (same units as z_axis)
    #      (not needed if solving for the whole trap)
    # returns the polynomial fit around the centre, assuming the z axis is zero at the centre
    roi_idxes = np.arange(np.argmax(z_axis > centre - roi_dist/2),
                          np.argmax(z_axis > centre + roi_dist/2))
    polys = interpolate_moments(z_axis-centre, moments, roi_idxes)
    return polys

def solve_scaled_constraints(moments, desired_pot, offset, scale_weight):
    # All array variables should be defined over the same set of x
    # coordinates (which are arbitrary)
    #
    # moments: electrode moments, M x N array, M = x coordinates,
    # N = number of (relevant)electrodes
    #
    # desired_pot: desired potential well kernel, which will be
    # scaled as large as electrode limits can support
    #
    # offset: potential offset required after scaling
    # scale_weight: how heavily to weight the scaling vs the fitting
    # Returns a column vector (2d) of the optimal electrode values
    states = []
    uopt = cvy.Variable(moments.shape[1], 1)
    vscale = cvy.Variable()
    cost = cvy.sum_squares(moments*uopt
                           - desired_pot*vscale - offset) # * vscale previously
    constr = [-max_elec_voltages[0] <= uopt, uopt <= max_elec_voltages[0]] # should make non-hardcoded
    # constr += [0.1 <= vscale]
    states.append(cvy.Problem(cvy.Minimize(cost-scale_weight*vscale), constr))
    prob = sum(states)        
    prob.solve(solver='ECOS', verbose=False)
    return uopt.value, vscale.value

def solve_poly_ab(poly_moments, alpha=0, slope_offset=None, dc_offset=None,
                  print_voltages=False, enforce_z_symmetry=False,
                  verbose_solver=False):
    # slope_offset: extra slope (electric field) to apply along z
    # direction, in V/m (can read it right off the potential
    # plots)
    #
    # dc_offset: extra potential to apply, in V (can read it right
    # off the plots)
    #
    # alpha: quadratic curvature (note: if the magnitude is too
    # large, the solver may fail)
    num_elec = poly_moments.shape[1]
    uopt = cvy.Variable(num_elec,1)
    # Quadratic and quartic terms in poly approximations
    alph_c = poly_moments[2,:]
    beta_c = poly_moments[0,:]
    gamm_c = poly_moments[3,:]
    dc_c = poly_moments[4,:]
    # for some reason normalisation is needed
    alph_norm = np.min(np.abs(alph_c))
    beta_norm = np.min(np.abs(beta_c))
    gamm_norm = np.min(np.abs(gamm_c))
    dc_norm = np.min(np.abs(dc_c))
    alph_co = alph_c/alph_norm
    beta_co = beta_c/beta_norm
    gamm_co = gamm_c/gamm_norm
    dc_co = dc_c/dc_norm
    # Avoid going over-voltage (don't quite hit the limits)
    constr = [-max_elec_voltages[0]+0.5 <= uopt, uopt <= max_elec_voltages[0]-0.5] 

    # electrode constraint pairs assume electrode moments are
    # adjacent, in order of increasing z and with the bottom row
    # following the top row. Additionally electrodes must be
    # symmetric around splitting zone.

    # Ensure x-y symmetric pairs of electrodes agree 
    for k in range(num_elec//2):
        constr.append(uopt[k] == uopt[k+num_elec//2])
    # Ensure z-symmetric (around splitting zone) pairs of electrodes agree
    if enforce_z_symmetry:
        for m in range(num_elec//4):
            constr.append(uopt[m] == uopt[-m-1])
    obj = cvy.Maximize(cvy.sum_entries(beta_co*uopt))
    obj -= 100*cvy.Minimize(cvy.sum_squares(alph_co*uopt - alpha/alph_norm))
    # constr.append(cvy.sum_entries(alph_co*uopt)==alpha/alph_norm) # quadratic term
    if dc_offset is not None:
        constr.append(cvy.sum_entries(dc_co*uopt)==dc_offset/dc_norm) # linear term ~= 0
    if slope_offset is not None:
        # obj -= cvy.Minimize(cvy.sum_squares(gamm_co*uopt - slope_offset/gamm_norm))
        # polyder_moments = np.vstack((np.polyder(k) for k in poly_moments.T)).T
        obj -= 10*cvy.Minimize(cvy.sum_squares(gamm_co*uopt - slope_offset/gamm_norm))
        
    # obj = cvy.Maximize(cvy.sum_entries(beta_co*uopt))+cvy.Maximize(-cvy.sum_entries(alph_co*uopt))
    prob = cvy.Problem(obj, constr)
    prob.solve(solver=global_solver, verbose=verbose_solver)
    ans = uopt.value
    if print_voltages:
        print("Voltages: ", uopt.value[:num_elec//2])
    assert ans is not None, "cvxpy did not find a solution."
    return ans, np.sum(alph_co*ans)*alph_norm, np.sum(beta_co*ans)*beta_norm

def merge_waveforms_for_rev(wfs):
    """Combines a list of waveforms into a single one that plays the list
    forward, then backward."""
    samples_forward = np.hstack(wf.samples for wf in wfs)
    samples_for_rev = np.hstack([samples_forward, np.fliplr(samples_forward)])
    wf_for_rev = Waveform("forward, then reverse", 0, "", samples_for_rev)
    wf_for_rev.set_new_uid()
    return wf_for_rev

def split_waveforms_many_resamples(
        start_loc, start_f, start_offset,
        final_locs, final_fs, final_offsets,
        split_loc, split_f, split_offset=None,
        n_transport=2000,
        electrode_subset=None,
        start_split_label='trans from start -> split start',
        split_label='split apart',
        plot_splits=False):
    # Specify the starting well properties (the experimental zone
    # usually) and the splitting well properties, which will be used
    # in the linear ramp between the combined well and the first stage
    # of the quartic polynomial solver.
    # Note: final_locs, final_freqs, final_offsets
    split_centre = split_loc*um # centre of the central splitting electrode moment
    polyfit_range = 200*um # spatial extent over which to fit polys to the solver

    # Prepare poly approximation in splitting zone
    polys = generate_interp_polys(trap_mom.transport_axis,
                                     trap_mom.potentials[:, electrode_subset],
                                     split_centre, polyfit_range)
    
    # Data format is (alpha, slope, points from prev. state to this one, interpolation function)
    # Requires careful tuning (TODO: automate it!)
    glob_sl_offs = 15
    interp_steps = 75

    # Generally go from positive alpha (start point is set by
    # start_loc, start_f, start_offset) to negative alpha.
    split_params = [# (1.5e7, None, 500, np.linspace),
        # (1e6, None, 500, np.linspace),
        #(0, glob_sl_offs, 500, lambda a,b,n: erfspace(a,b,n,1.5)),
        #        (1e6, glob_sl_offs, 200, np.linspace), # TODO: uncomment this
        (0, glob_sl_offs, interp_steps, np.linspace),
        # (-3e6, None, 500, np.linspace),
        (-5e6, glob_sl_offs, interp_steps, np.linspace),
        (-1e7, glob_sl_offs, interp_steps, np.linspace),
        (-1.5e7, glob_sl_offs, interp_steps, np.linspace)]

    # (-2e7, None, 50, np.linspace),
    # (-3e7, None, interp_steps, np.linspace),
    # (-4e7, None, interp_steps, np.linspace),
    # (-5e7, None, 150, np.linspace),
    # (-6e7, None, 300, np.linspace)]

    if not split_offset:
        # automatically figure out the potential offset by running the
        # solver for the initial splitting conditions and fitting to it
        death_v_set = np.zeros([num_elecs, 1])
        sp_start = split_params[0]
        elec_v_set,_,_ = solve_poly_ab(polys, sp_start[0], sp_start[1])
        death_v_set[physical_electrode_transform[electrode_subset]] = elec_v_set
        wavpot_fit = find_wells_from_samples(death_v_set,
                                             roi_centre=split_centre,
                                             roi_width=polyfit_range)
        assert len(wavpot_fit['offsets']) == 1, "Error, found too many wells in ROI at start of splitting."
        split_offset = wavpot_fit['offsets'][0]/meV
        
    # Initial waveform, transports from start to splitting location
    wf_split = tu.transport_waveform(
        [start_loc, split_loc],
        [start_f, split_f],
        [start_offset, split_offset], n_transport, start_split_label)
    
    latest_death_voltages = wf_split.samples[:,[-1]] # square bracket to return column vector
    full_wfm_voltages = latest_death_voltages.copy()

    debug_splitting_parts = False
    # Prepare full voltage array
    for (alpha, slope_offset, npts, linspace_fn) in split_params:
        elec_voltage_set,alpha,beta = solve_poly_ab(polys, alpha,
                                                    slope_offset=slope_offset, dc_offset=None)
        new_death_voltages = latest_death_voltages.copy()
        new_death_voltages[physical_electrode_transform[electrode_subset]] = elec_voltage_set

        # Ramp from old to new voltage set
        ramped_voltages = vlinspace(latest_death_voltages, new_death_voltages,
                                    npts, linspace_fn)[:,1:]
        full_wfm_voltages = np.hstack([full_wfm_voltages, ramped_voltages])
        latest_death_voltages = new_death_voltages

        # st()
        
        if debug_splitting_parts:
            new_wf = Waveform("", 0, "", ramped_voltages)
            asdf = WavPotential(new_wf)
            asdf.plot_range_of_wfms(20)
            plt.show()

    final_splitting_params = find_wells_from_samples(
        latest_death_voltages, split_centre, polyfit_range)
    split_locs = np.array(final_splitting_params['locs'])/um
    split_freqs = np.array(final_splitting_params['freqs'])/MHz
    split_offsets = np.array(final_splitting_params['offsets'])/meV
    assert split_locs.size == 2, "Wrong number of wells detected after splitting"

                                                                     
    # Final waveform, extends separation by 150um either way and goes to default well settings
    # (starting values must be set to the results of the splitting!)
    wf_finish_split = tu.transport_waveform_multiple(
        [[split_locs[0], final_locs[0]],[split_locs[1], final_locs[1]]],
        [[split_freqs[0],final_fs[0]],[split_freqs[1],final_fs[1]]],
        [[split_offsets[0], final_offsets[0]],[split_offsets[1], final_offsets[1]]],
        n_transport,
        "")
    
    # Remove final segment of full voltage array, replace with manual
    # ramp to start of regular solver
    final_ramp_start = full_wfm_voltages[:,[-npts]]
    final_ramp_end = wf_finish_split.samples[:,[0]]
    full_wfm_voltages = full_wfm_voltages[:,:-npts+1] # final_ramp_start voltage set

    final_ramped_voltages = vlinspace(final_ramp_start, final_ramp_end, npts, linspace_fn)[:,1:]
    full_wfm_voltages = np.hstack([full_wfm_voltages, final_ramped_voltages])

    # Append final splitting wfm
    full_wfm_voltages = np.hstack([full_wfm_voltages, wf_finish_split.samples[:,1:]])
    
    splitting_wf = Waveform(split_label, 0, "", full_wfm_voltages)
    splitting_wf.set_new_uid()
    
    if False:
        asdf = WavPotential(splitting_wf)
        print(asdf.find_wells(-1))
        asdf.plot_one_wfm(-1)
        plt.show()
    animate_waveform = False
    if animate_waveform:
        # Set up formatting for the movie files
        Writer = anim.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_ylim([-4,4])
        line, = ax.plot(asdf.trap_axis/um, asdf.potentials[:,0])
        def update(data):
            line.set_ydata(data)
            return line
        
        def data_gen():
            for pot in asdf.potentials.T[::10]:
                yield pot

        # im_ani = anim.ArtistAnimation(plt.figure(), ims, interval=100, repeat_delay=5000, blit=True)
        
        im_ani = anim.FuncAnimation(fig, update, data_gen, interval=30)

        plt.show()
        # im_ani.save('im.mp4', writer=writer)

    def lin_gen(a, b, npts, erf_sc):
        return erfspace(a, b, npts, erf_sc)
        
    wf_split_tup = tuple(tu.transport_waveform(
        [start_loc, split_loc],
        [start_f, split_f],
        # [start_offset, split_offset],
        [start_offset, start_offset+d],
        n_transport,
        start_split_label+'extra ' + str(d),
        linspace_fn = lambda a, b, npts: erfspace(a, b, npts, 2.5))

                         for d in np.linspace(-860,-660,16))

    return wf_split, splitting_wf, wf_split_tup

if __name__ == "__main__":
    #reproduce_fig2_home_steane()
    # look_at_wells_manually()
    wp = WaveformSet(waveform_file="waveform_files/load_split_2Be2Ca_2016_07_06_v04.dwc.json").find_waveform("apart").samples[:,[50]]
    find_coulomb_wells(wp, -422.5*um, 100*um, plot_results=True)
    # wpot = WavPotential(

    
    
    # wp = WaveformSet(waveform_file = "waveform_files/load_split_2Be2Ca_2016_07_06_v04.dwc.json").find_waveform("static")

    # wpot = WavPotential(wp, ion_mass=mass_Ca)
    # wpot.plot_radials(0)
    # plt.show()




#### Splitting routine used right now for most waveforms (originally from load_and_split.py)    

def split_waveforms(
        start_loc, start_f, start_offset,
        final_locs, final_fs, final_offsets,
        split_loc, split_f, split_offset=None,
        n_transport=2000,        
        field_offset = 0,
        electrode_subset=None,
        start_split_label='trans from start -> split start',
        split_label='split apart',
        plot_splits=False):
    # Specify the starting well properties (the experimental zone
    # usually) and the splitting well properties, which will be used
    # in the linear ramp between the combined well and the first stage
    # of the quartic polynomial solver.
    # Note: final_locs, final_freqs, final_offsets
    split_centre = split_loc*um # centre of the central splitting electrode moment
    polyfit_range = 200*um

    polys = generate_interp_polys(trap_mom.transport_axis,
                                    trap_mom.potentials[:, electrode_subset],
                                    split_centre, polyfit_range)
    
    # Data format is (alpha, slope, points from prev. state to this one)
    # Requires careful tuning

    # field_offset = -33.5 # 2Ca splitting
    # field_offset = -35
    # field_offset = -10 # 2Be splitting
    
    interp_steps = 75
    split_params = [# (1.5e7, None, 500, np.linspace),
        # (1e6, None, 500, np.linspace),
        #(0, field_offset, 500, lambda a,b,n: erfspace(a,b,n,1.5)),
#        (1e6, field_offset, 200, np.linspace), # TODO: uncomment this
#        (1e6, field_offset, interp_steps//2, np.linspace),        
        (5e5, field_offset, interp_steps, np.linspace),
        (0, field_offset, interp_steps//2, np.linspace),
        # (-3e6, None, 500, np.linspace),
        (-2.5e6, field_offset, interp_steps//2, np.linspace),
        (-5e6, field_offset, interp_steps//2, np.linspace),
        (-7.5e6, field_offset, interp_steps//2, np.linspace),        
        (-1e7, field_offset, interp_steps//2, np.linspace),
        (-1.5e7, field_offset, interp_steps//2, np.linspace)]
        # (-2e7, None, 50, np.linspace),
        # (-3e7, None, interp_steps, np.linspace),
        # (-4e7, None, interp_steps, np.linspace),
        # (-5e7, None, 150, np.linspace),
        # (-6e7, None, 300, np.linspace)]

    if not split_offset:
        # automatically figure out the potential offset by running the
        # solver for the initial splitting conditions and fitting to it
        death_v_set = np.zeros([num_elecs, 1])
        sp_start = split_params[0]
        elec_v_set,_,_ = solve_poly_ab(polys, alpha=sp_start[0],
                                          slope_offset=sp_start[1])
        death_v_set[physical_electrode_transform[electrode_subset]] = elec_v_set
        wavpot_fit = find_wells_from_samples(death_v_set,
                                             roi_centre=split_centre,
                                             roi_width=polyfit_range)
        assert len(wavpot_fit['offsets']) == 1, "Error, found too many wells in ROI at start of splitting."
        split_offset = wavpot_fit['offsets'][0]/meV
        
    # Initial waveform, transports from start to splitting location
    wf_split = tu.transport_waveform(
        [start_loc, split_loc],
        [start_f, split_f],
        [start_offset, split_offset], n_transport, start_split_label,
        interp_start=20)
        
    latest_death_voltages = wf_split.samples[:,[-1]] # square bracket to return column vector
    full_wfm_voltages = latest_death_voltages.copy()

    plot_splitting_parts = False
    # Prepare full voltage array
    for (alpha, slope_offset, npts, linspace_fn) in split_params:
        elec_voltage_set,alpha,beta = solve_poly_ab(polys, alpha,
                                            slope_offset=slope_offset, dc_offset=None)
        new_death_voltages = latest_death_voltages.copy()
        new_death_voltages[physical_electrode_transform[electrode_subset]] = elec_voltage_set

        # Ramp from old to new voltage set
        ramped_voltages = vlinspace(latest_death_voltages, new_death_voltages,
                                    npts, linspace_fn)[:,1:]
        full_wfm_voltages = np.hstack([full_wfm_voltages, ramped_voltages])
        latest_death_voltages = new_death_voltages
        
        if plot_splitting_parts:
            new_wf = Waveform("", 0, "", ramped_voltages)
            asdf = WavPotential(new_wf)
            asdf.plot_range_of_wfms(20)
            # plt.set_ylim([-1.470,-1.435])
            # plt.set_xlim([-500,-350])            
            plt.show()

    npts_final = npts

    final_splitting_params = find_wells_from_samples(
        latest_death_voltages, split_centre, polyfit_range)
    split_locs = np.array(final_splitting_params['locs'])/um
    split_freqs = np.array(final_splitting_params['freqs'])/MHz
    split_offsets = np.array(final_splitting_params['offsets'])/meV
    assert split_locs.size == 2, "Wrong number of wells detected after splitting"

                                                                     
    # Final waveform, extends separation by 150um either way and goes to default well settings
    # (starting values must be set to the results of the splitting!)
    wf_finish_split = tu.transport_waveform_multiple(
        [[split_locs[0], final_locs[0]],[split_locs[1], final_locs[1]]],
        [[split_freqs[0],final_fs[0]],[split_freqs[1],final_fs[1]]],
        [[split_offsets[0], final_offsets[0]],[split_offsets[1], final_offsets[1]]],
        n_transport,
        "")
    
    # Remove final segment of full voltage array, replace with manual
    # ramp to start of regular solver
    final_ramp_start = full_wfm_voltages[:,[-npts_final]]
    final_ramp_end = wf_finish_split.samples[:,[0]]
    full_wfm_voltages = full_wfm_voltages[:,:-npts_final+1] # final_ramp_start voltage set

    final_ramped_voltages = vlinspace(final_ramp_start, final_ramp_end, npts_final*3//4, linspace_fn)[:,1:]
    full_wfm_voltages = np.hstack([full_wfm_voltages, final_ramped_voltages])

    # Append final splitting wfm
    full_wfm_voltages = np.hstack([full_wfm_voltages, wf_finish_split.samples[:,1:]])

    savgol_smooth = True
    # Smooth the waveform voltages
    if savgol_smooth:
        full_wfm_voltages_filt = ssig.savgol_filter(full_wfm_voltages, 151, 2, axis=-1)
    else:
        full_wfm_voltages_filt = full_wfm_voltages

    spline_smooth = False
    if spline_smooth:
        t_axis = np.arange(full_wfm_voltages.shape[1])        
        fwv_funcs = (sintp.UnivariateSpline(t_axis, fwv, s=10, k=5) for fwv in full_wfm_voltages)
        full_wfm_voltages_filt = np.vstack((fwv_func(t_axis) for fwv_func in fwv_funcs))
            
    # splitting_wf = Waveform(split_label, 0, "", full_wfm_voltages)
    splitting_wf = Waveform(split_label+", offset = " + "{0:6.3e}".format(field_offset*um) + " V/m",
                            0, "", full_wfm_voltages_filt)
    splitting_wf.set_new_uid()
    
    if False:
        asdf = WavPotential(splitting_wf)
        print(asdf.find_wells(-1))
        asdf.plot_one_wfm(-1)
        plt.show()
    animate_waveform = False
    if animate_waveform:
        # Set up formatting for the movie files
        Writer = anim.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_ylim([-4,4])
        line, = ax.plot(asdf.trap_axis/um, asdf.potentials[:,0])
        def update(data):
            line.set_ydata(data)
            return line
        
        def data_gen():
            for pot in asdf.potentials.T[::10]:
                yield pot

        # im_ani = anim.ArtistAnimation(plt.figure(), ims, interval=100, repeat_delay=5000, blit=True)
        
        im_ani = anim.FuncAnimation(fig, update, data_gen, interval=30)

        plt.show()
        im_ani.save('im.mp4', writer=writer)

    return wf_split, splitting_wf
