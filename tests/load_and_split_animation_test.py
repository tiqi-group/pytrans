#!/usr/bin/env python3

import sys
sys.path.append("../")
from pytrans import *

def animate_wfms(wfm_tup, decimation=10):
    # wfm_tup: must be a list of similar waveforms, whose samples matrices are the same size
    # decimation: factor reduction in sample number
    Writer = anim.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist="vnegnev"), bitrate=1800)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim([-4,4])
    lines = tuple(ax.plot(k.trap_axis/um, k.potentials[:,0])[0] for k in wfm_tup)
    # st()
    def data_gen(i):
        return tuple(wf.potentials[:, i*decimation] for wf in wfm_tup)
    
    def animate(i):
        ydata = data_gen(i)
        for line, y in zip(lines, ydata):
            line.set_ydata(y)

        #return list(l.set_ydata(d) for l, d in zip(lines, data))
        # st()
        #return lines

    animate(0)
        
    im_ani = anim.FuncAnimation(fig, animate,
                                frames = wfm_tup[0].potentials.shape[1]//decimation,
                                interval=30, blit=False)
    plt.show()

if __name__ == "__main__":
    wf_paths = []
    wf_paths.append(os.path.join(os.pardir, "waveform_files", "load_split_2016_06_23_v05.dwc.json"))
    wf_paths.append(os.path.join(os.pardir, "waveform_files", "load_split_2016_06_28_v01.dwc.json"))
    wf_paths.append(os.path.join(os.pardir, "waveform_files", "load_split_2016_06_28_v02.dwc.json"))
    wf_paths.append(os.path.join(os.pardir, "waveform_files", "load_split_2016_06_28_v03.dwc.json"))

    wfss = tuple(WaveformSet(waveform_file = wfp) for wfp in wf_paths) # waveform sets
    wfps = tuple(WavPotential(wfs.get_waveform(-1)) for wfs in wfss)

    # Animation style: 2 sequential Figures or one overlapped Figure
    series = False
    parallel = True
    
    if series:            
        wfps.animate_wfm(1)

    if parallel:
        animate_wfms(wfps, 1)


