# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Thu Feb 27 16:35:30 2025"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: Figure 7
"""

#-----------------------------------------------------------------------------#
# modules

import sys
sys.path.append(r'D:\File\Paper\Vibration\codes\cat')
import os

from copy import deepcopy

import numpy as np
import scipy as sp

import h5py as h5
import matplotlib.pyplot as plt
from cat.wave_optics.widget import gaussian_fit
from cat.wave_optics.optics import screen
from cat.wave_optics.widget import plot_optic

#-----------------------------------------------------------------------------#
# parameters

file_header = r"D:\File\Paper\Vibration\codes\sr_source\vibration"
svd_file1 = r"focus_svd_method_25nrad4.h5"
svd_file2 = r"focus_svd_method_0nrad4.h5"
svd_file3 = r"focus_svd_method_50nrad4.h5"

unit = '$\u03bc$m' 

time = np.linspace(0, 0.2, 1000)

#-----------------------------------------------------------------------------#
# functions

def center_index(wavefront, x_range, y_range):
    
    x_para, fitted = gaussian_fit(x_range, np.sum(np.abs(wavefront)**2, 0))
    y_para, fitted = gaussian_fit(y_range, np.sum(np.abs(wavefront)**2, 1))
    
    return [x_para[1], y_para[1]]

#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    #---------------------------------------------------
    # wavefront reconstruction
    
    file_svd_result1 = os.path.join(file_header, svd_file1)
    file_svd_result2 = os.path.join(file_header, svd_file2)
    file_svd_result3 = os.path.join(file_header, svd_file3)

    fs1 = h5.File(file_svd_result1, 'r')
    fs2 = h5.File(file_svd_result2, 'r')
    fs3 = h5.File(file_svd_result3, 'r')

    vib_list1, vib_list2, vib_list3 = list(), list(), list()

    #--------------------------------
    # svd 25 urad
    
    vibration_cmode1 = list()
    for idx in range(1000):
        
        i_optic_mode1 = np.zeros((512, 512), dtype = np.complex64)
        for i_mirror in range(10):
            i_optic_mode1 += (
                np.array(fs1["coherence/evolution"][i_mirror, idx]) * 
                np.array(fs1["coherence/ratio"][i_mirror]) * 
                np.array(fs1["coherence/coherent_mode"][i_mirror, :, :])
                )
        
        vibration_cmode1.append(i_optic_mode1)
        vibration_cmode1.append(i_optic_mode1)
        vibx_idx, viby_idx = center_index(
            i_optic_mode1, 
            np.linspace(-996, 1004, 512), np.linspace(-996, 1004, 512)
            )
        vib_list1.append([vibx_idx, viby_idx])
        
    #--------------------------------
    # svd 0 urad
    
    vibration_cmode2 = list()
    for idx in range(1000):
        
        i_optic_mode2 = np.zeros((512, 512), dtype = np.complex64)
        for i_mirror in range(10):
            i_optic_mode2 += (
                np.array(fs2["coherence/evolution"][i_mirror, idx]) * 
                np.array(fs2["coherence/ratio"][i_mirror]) * 
                np.array(fs2["coherence/coherent_mode"][i_mirror, :, :])
                )
        
        vibration_cmode2.append(i_optic_mode2)
        vibration_cmode2.append(i_optic_mode2)
        vibx_idx, viby_idx = center_index(
            i_optic_mode2, 
            np.linspace(-996, 1004, 512), np.linspace(-996, 1004, 512)
            )
        vib_list2.append([vibx_idx, viby_idx])
        
    #--------------------------------
    # svd 50 urad
    
    vibration_cmode3 = list()
    for idx in range(1000):
        
        i_optic_mode3 = np.zeros((512, 512), dtype = np.complex64)
        for i_mirror in range(10):
            i_optic_mode3 += (
                np.array(fs3["coherence/evolution"][i_mirror, idx]) * 
                np.array(fs3["coherence/ratio"][i_mirror]) * 
                np.array(fs3["coherence/coherent_mode"][i_mirror, :, :])
                )
        
        vibration_cmode3.append(i_optic_mode3)
        vibration_cmode3.append(i_optic_mode3)
        vibx_idx, viby_idx = center_index(
            i_optic_mode3, 
            np.linspace(-996, 1004, 512), np.linspace(-996, 1004, 512)
            )
        vib_list3.append([vibx_idx, viby_idx])
    
    #---------------------------------------------------
    # coherence ratio calculation
    
    focus_svd_test1 = screen(optic_file = file_svd_result1, n_vector = 1)
    focus_svd_test1.n_vector = 1000
    focus_svd_test1.n = 1000
    focus_svd_test1.ratio = np.ones(1000)
    focus_svd_test1.cmode = deepcopy(vibration_cmode1)
    focus_svd_test1.decomposition(method = "sparse_svd", sparse_n = 50)
    focus_svd_test1.n_vector = 50
    focus_svd_test1.n = 50
    svd_ratio1 = np.abs(focus_svd_test1.ratio[0 : 15])**2 / np.sum(
        np.abs(focus_svd_test1.ratio[0 : 50])**2
        )
    
    focus_svd_test2 = screen(optic_file = file_svd_result2, n_vector = 1)
    focus_svd_test2.n_vector = 1000
    focus_svd_test2.n = 1000
    focus_svd_test2.ratio = np.ones(1000)
    focus_svd_test2.cmode = deepcopy(vibration_cmode2)
    focus_svd_test2.decomposition(method = "svd") # require high-accuary
    focus_svd_test2.n_vector = 50
    focus_svd_test2.n = 50
    svd_ratio2 = np.abs(focus_svd_test2.ratio[0 : 15])**2 / np.sum(
        np.abs(focus_svd_test2.ratio[0 : 50])**2
        )
    
    focus_svd_test3 = screen(optic_file = file_svd_result3, n_vector = 1)
    focus_svd_test3.n_vector = 1000
    focus_svd_test3.n = 1000
    focus_svd_test3.ratio = np.ones(1000)
    focus_svd_test3.cmode = deepcopy(vibration_cmode3)
    focus_svd_test3.decomposition(method = "svd") # require high-accuary
    focus_svd_test3.n_vector = 50
    focus_svd_test3.n = 50
    svd_ratio3 = np.abs(focus_svd_test3.ratio[0 : 15])**2 / np.sum(
        np.abs(focus_svd_test3.ratio[0 : 50])**2
        )
        
    #---------------------------------------------------
    # plot
    
    figure, axes = plt.subplots(2, 4, figsize = (12, 6))
    
    #--------------------------------
    # different vibration modes
    
    axes[0, 0].plot(time, np.array(vib_list3)[:, 0], linewidth = 3, alpha = 0.5)
    axes[0, 0].plot(time, np.array(vib_list1)[:, 0], linewidth = 3, alpha = 0.5)
    axes[0, 0].plot(time, np.array(vib_list2)[:, 0], linewidth = 3, alpha = 0.5)
    
    axes[0, 0].set_xlabel('time (s)', fontsize = 12)
    axes[0, 0].set_ylabel('x (nm)', fontsize = 12)
    axes[0, 0].yaxis.set_label_coords(-0.26, 0.5)
    axes[0, 0].set_box_aspect(1)
    
    axes[0, 1].plot(time, np.array(vib_list3)[:, 1], linewidth = 3, alpha = 0.5)
    axes[0, 1].plot(time, np.array(vib_list1)[:, 1], linewidth = 3, alpha = 0.5)
    axes[0, 1].plot(time, np.array(vib_list2)[:, 1], linewidth = 3, alpha = 0.5)

    axes[0, 1].set_xlabel('time (s)', fontsize = 12)
    axes[0, 1].set_ylabel('y (nm)', fontsize = 12)
    axes[0, 1].yaxis.set_label_coords(-0.26, 0.5)
    axes[0, 1].set_box_aspect(1)
    
    axes[0, 2].scatter(range(5), svd_ratio3[0 : 5])
    axes[0, 2].plot(range(5), svd_ratio3[0 : 5], linewidth = 3, alpha = 0.5)
    axes[0, 2].scatter(range(5), svd_ratio1[0 : 5])
    axes[0, 2].plot(range(5), svd_ratio1[0 : 5], linewidth = 3, alpha = 0.5)
    axes[0, 2].scatter(range(5), svd_ratio2[0 : 5])
    axes[0, 2].plot(range(5), svd_ratio2[0 : 5], linewidth = 3, alpha = 0.5)
    
    axes[0, 2].set_xlabel('time (s)', fontsize = 12)
    axes[0, 2].set_ylabel('coherent ratio', fontsize = 12)
    axes[0, 2].yaxis.set_label_coords(-0.26, 0.5)
    axes[0, 2].set_box_aspect(1)   
    
    figure.tight_layout()
    plt.savefig(r"D:\File\Paper\Vibration\codes\sr_source\vibration\figure7.png", dpi = 1000)
    
    
    
