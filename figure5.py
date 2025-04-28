# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Tue Jan 21 23:33:52 2025"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: Figure 5
"""

#-----------------------------------------------------------------------------#
# modules


import sys
sys.path.append(r'D:\File\Paper\Vibration\codes\cat')
import os
import time as t

from copy import deepcopy

import numpy as np
import scipy as sp

import h5py as h5
import matplotlib.pyplot as plt

from cat.wave_optics.optics import source_optic, screen, kb
from cat.wave_optics.propagate import fresnel, czt
from cat.wave_optics._vibration import sinc_vibration

#-----------------------------------------------------------------------------#
# parameters

file_header = r"D:\File\Paper\Vibration\codes\sr_source"
source_file = os.path.join(file_header, r"source\b4_pdr_12400eV.h5")
wavefront_file = os.path.join(file_header, r"source\b4_12400eV_se.npy")

unit = '$\u03bc$m' 
mode_label0 = '$\u03C6$\u2080'

wavefront = np.load(wavefront_file)
wavefront = np.abs(wavefront) * np.exp(-1j * np.angle(wavefront))

#--------------------------------------------------------------------
# parameters for vibation

#---------------------------------------------------------
# time and vibration parameters

time_sr = np.linspace(0, 0.01, 100)
time_dcm = np.linspace(0, 0.05, 100)
time_kb = np.linspace(0, 0.1, 100)
time = np.linspace(0, 0.2, 1000)

# sr source

not_sr_vibration = False
flag = 0 if not_sr_vibration else 1

sr_amp_rotx = 3 * [flag * 0.30e-6 * 1.414 / np.sqrt(3)]
sr_phase_rotx = [0.11 * np.pi, -0.33 * np.pi, 0.57 * np.pi]
sr_freq_rotx = [100, 200, 500]

sr_amp_roty = 3 * [flag * 0.12e-6 * 1.414 / np.sqrt(3)]
sr_phase_roty = [0.27 * np.pi, 0.73 * np.pi, -0.91 * np.pi]
sr_freq_roty = [100, 200, 500]

sr_amp_offx = 3 * [flag * 0.88e-6 * 1.414 / np.sqrt(3)]
sr_phase_offx = [0.13 * np.pi, 0.25 * np.pi, 0.55 * np.pi]
sr_freq_offx = [100, 200, 500]

sr_amp_offy = 3 * [flag * 0.23e-6 * 1.414 / np.sqrt(3)]
sr_phase_offy = [0.23 * np.pi, 1.51 * np.pi, 1.71 * np.pi]
sr_freq_offy = [100, 200, 500]

# hdcm

not_dcm_vibration = False
flag = 0 if not_dcm_vibration else 1

dcm_amp_rotx = 3 * [flag * 50.0e-09 * 1.414 / np.sqrt(3)]
dcm_phase_rotx = [0.11 * np.pi, 0.57 * np.pi, 1.71 * np.pi]
dcm_freq_rotx = [20, 50, 100]

# kb mirror

use_akb_vibration = False
flag = 0 if use_akb_vibration else 1

kb_amp_rotx = 3 * [flag * 50.0e-09 * 1.414 / np.sqrt(3)]
kb_phase_rotx = [0.47 * np.pi, 0.11 * np.pi, -0.07 * np.pi]
kb_freq_rotx = [10, 20, 30]

kb_amp_roty = 3 * [flag * 50.0e-09 * 1.414 / np.sqrt(3)]
kb_phase_roty = [0.73 * np.pi, 1.11 * np.pi, -0.26 * np.pi]
kb_freq_roty = [10, 20, 30]

#---------------------------------------------------------
# all the vibration

sr_rotx_list = sinc_vibration(sr_amp_rotx, sr_freq_rotx, sr_phase_rotx, time)
sr_roty_list = sinc_vibration(sr_amp_roty, sr_freq_roty, sr_phase_roty, time)
sr_offx_list = sinc_vibration(sr_amp_offx, sr_freq_offx, sr_phase_offx, time)
sr_offy_list = sinc_vibration(sr_amp_offy, sr_freq_offy, sr_phase_offy, time)

dcm_rotx_list = sinc_vibration(
    dcm_amp_rotx, dcm_freq_rotx, dcm_phase_rotx, time
    )
kb_rotx_list = sinc_vibration(kb_amp_rotx, kb_freq_rotx, kb_phase_rotx, time)
kb_roty_list = sinc_vibration(kb_amp_roty, kb_freq_roty, kb_phase_roty, time)

#-----------------------------------------------------------------------------#
# parameters

#-----------------------------------------------------------------------------#
# functions

#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    #---------------------------------------------------
    # plot
    
    figure, axes = plt.subplots(1, 2, figsize = (12, 3))
    
    axes[0].plot(time, np.array(sr_rotx_list) * 1e9, linewidth = 3, alpha = 0.5)
    axes[0].plot(time, np.array(sr_roty_list) * 1e9, linewidth = 3, alpha = 0.5)
    axes[0].plot(time, np.array(dcm_rotx_list) * 1e9, linewidth = 3, alpha = 0.5)
    axes[0].plot(time, np.array(kb_rotx_list) * 1e9, linewidth = 3, alpha = 0.5)
    axes[0].plot(time, np.array(kb_roty_list) * 1e9, linewidth = 3, alpha = 0.5)
    axes[0].set_title("angle vibration")
    axes[0].set_xlim((-0.005, 0.205))
    axes[0].set_xlabel('time (s)', fontsize = 12)
    axes[0].set_ylabel('angle (nrad)', fontsize = 12)
    axes[0].yaxis.set_label_coords(-0.1, 0.5)
    
    axes[1].plot(time, np.array(sr_offx_list) * 1e9, linewidth = 3, alpha = 0.5)
    axes[1].plot(time, np.array(sr_offy_list) * 1e9, linewidth = 3, alpha = 0.5)
    axes[1].set_title("position vibration")
    axes[1].set_xlim((-0.005, 0.205))
    axes[1].set_xlabel('time (s)', fontsize = 12)
    axes[1].set_ylabel('position (nm)', fontsize = 12)
    axes[1].yaxis.set_label_coords(-0.1, 0.5)
    
    figure.tight_layout()
    plt.savefig(r"D:\File\Paper\Vibration\codes\sr_source\vibration\figure5.png", dpi = 1000)
    
