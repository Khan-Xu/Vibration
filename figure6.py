# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Sun Feb 23 21:42:50 2025"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: Figure 6 a-h plot
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
propagation_file = r"HEPS_B4_KBfocus_propagation_method_50nrad.h5"
svd_file = r"HEPS_B4_KBfocus_svd_method_50nrad.h5"

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
    
    file_propagate = os.path.join(file_header, propagation_file)
    file_svd_result = os.path.join(file_header, svd_file)

    fp = h5.File(file_propagate, 'r')
    fs = h5.File(file_svd_result, 'r')

    vib_list, tro_list = list(), list()
    
    #--------------------------------
    # propagation
    
    propagation_cmode = list()
    
    for idx in range(1000):
        
        i_optic_mode = np.zeros((512, 512), dtype = np.complex64)
        for i_mirror in range(10):
            i_optic_mode += (
                np.array(fp["coherence/evolution"][i_mirror, idx]) * 
                np.array(fp["coherence/ratio"][i_mirror]) * 
                np.array(fp["coherence/coherent_mode"][i_mirror, :, :])
                )
        propagation_cmode.append(i_optic_mode)
        trox_idx, troy_idx = center_index(
            i_optic_mode, 
            np.linspace(-996, 1004, 512), np.linspace(-996, 1004, 512)
            )
        tro_list.append([trox_idx, troy_idx])

    #--------------------------------
    # svd
    
    vibration_cmode = list()
    for idx in range(1000):
        
        i_optic_mode = np.zeros((512, 512), dtype = np.complex64)
        for i_mirror in range(10):
            i_optic_mode += (
                np.array(fs["coherence/evolution"][i_mirror, idx]) * 
                np.array(fs["coherence/ratio"][i_mirror]) * 
                np.array(fs["coherence/coherent_mode"][i_mirror, :, :])
                )
        
        vibration_cmode.append(i_optic_mode)
        vibration_cmode.append(i_optic_mode)
        vibx_idx, viby_idx = center_index(
            i_optic_mode, 
            np.linspace(-996, 1004, 512), np.linspace(-996, 1004, 512)
            )
        vib_list.append([vibx_idx, viby_idx])
    
    #---------------------------------------------------
    # coherence ratio calculation
    
    focus_svd_test = screen(optic_file = file_svd_result, n_vector = 1)
    focus_svd_test.n_vector = 1000
    focus_svd_test.n = 1000
    focus_svd_test.ratio = np.ones(1000)
    focus_svd_test.cmode = deepcopy(vibration_cmode)
    focus_svd_test.decomposition(method = "sparse_svd", sparse_n = 50)
    focus_svd_test.n_vector = 50
    focus_svd_test.n = 50
    svd_ratio = np.abs(focus_svd_test.ratio[0 : 15])**2 / np.sum(
        np.abs(focus_svd_test.ratio[0 : 50])**2
        )
    
    focus_propagate_test = screen(optic_file = file_propagate, n_vector = 1)
    focus_propagate_test.n_vector = 1000
    focus_propagate_test.n = 1000
    focus_propagate_test.ratio = np.ones(1000)
    focus_propagate_test.cmode = deepcopy(propagation_cmode)
    focus_propagate_test.decomposition(method = "sparse_svd", sparse_n = 50)
    focus_propagate_test.n_vector = 50
    focus_propagate_test.n = 50
    tro_ratio = (
        np.abs(focus_propagate_test.ratio[0 : 15])**2 / 
        np.sum(np.abs(focus_propagate_test.ratio[0 : 50])**2)
        )
        
    #---------------------------------------------------
    # plot
    
    figure, axes = plt.subplots(4, 4, figsize = (12, 12))
    
    #--------------------------------
    # different vibration modes
    
    for idx in range(4):
        
        axes[0, idx].imshow(
            np.abs(np.array(fs["coherence/coherent_mode"][idx, 256 - 128 : 256 + 128, 256 - 128 : 256 + 128])**2), 
            extent = [-496, 504, -496, 504]
            )
        axes[0, idx].set_xlabel('x (nm)', fontsize = 12)
        axes[0, idx].set_ylabel('y (nm)', fontsize = 12)
        axes[0, idx].yaxis.set_label_coords(-0.26, 0.5)
        
        axes[1, idx].imshow(
            np.abs(np.array(fp["coherence/coherent_mode"][idx, 256 - 128 : 256 + 128, 256 - 128 : 256 + 128])**2), 
            extent = [-496, 504, -496, 504]
            )
        axes[1, idx].set_xlabel('x (nm)', fontsize = 12)
        axes[1, idx].set_ylabel('y (nm)', fontsize = 12)
        axes[1, idx].yaxis.set_label_coords(-0.26, 0.5)
    
    axes[2, 0].plot(np.array(vib_list)[:, 0], np.array(vib_list)[:, 1], linewidth = 3, alpha = 0.5)
    axes[2, 0].plot(np.array(tro_list)[:, 0], np.array(tro_list)[:, 1], linewidth = 3, alpha = 0.5)
    axes[2, 0].set_xlim([-60, 60])
    axes[2, 0].set_ylim([-80, 100])
    axes[2, 0].set_xlabel('x (nm)', fontsize = 12)
    axes[2, 0].set_ylabel('y (nm)', fontsize = 12)
    axes[2, 0].yaxis.set_label_coords(-0.26, 0.5)
    axes[2, 0].set_box_aspect(1)
    
    axes[2, 1].scatter(range(10), svd_ratio[0 : 10], alpha = 0.5)
    axes[2, 1].plot(range(10), svd_ratio[0 : 10], linewidth = 3, alpha = 0.5)
    axes[2, 1].scatter(range(10), tro_ratio[0 : 10], alpha = 0.5)
    axes[2, 1].plot(range(10), tro_ratio[0 : 10], linewidth = 3, alpha = 0.5)
    
    axes[2, 1].set_xlabel('index (n)', fontsize = 12)
    axes[2, 1].set_ylabel('coherence ratio', fontsize = 12)
    axes[2, 1].yaxis.set_label_coords(-0.26, 0.5)
    axes[2, 1].set_box_aspect(1)
    
    # # for idx in range(2):
    
    # vib_parameter_s = np.abs(np.array(fs["coherence/evolution"][0, :]))
    # vib_parameter_p = np.abs(np.array(fp["coherence/evolution"][0, :]))        
    # axes[2, 2].plot(time, vib_parameter_s, linewidth = 3, alpha = 0.5) 
    # axes[2, 2].plot(time, vib_parameter_p, linewidth = 3, alpha = 0.5)
    
    # vib_parameter_s = np.abs(np.array(fs["coherence/evolution"][1, :]))
    # vib_parameter_p = np.abs(np.array(fp["coherence/evolution"][1, :]))        
    # axes[2, 2].plot(time, vib_parameter_s, linewidth = 3, alpha = 0.5) 
    # axes[2, 2].plot(time, vib_parameter_p, linewidth = 3, alpha = 0.5)
    
    # axes[2, 2].set_xlabel('time (s)', fontsize = 12)
    # axes[2, 2].set_ylabel('vibration parameters (a. u.)', fontsize = 12)
    # axes[2, 2].yaxis.set_label_coords(-0.26, 0.5)
    # axes[2, 2].set_box_aspect(1)
    
    vib_parameter_s = np.abs(np.array(fs["coherence/evolution"][2, :]))
    vib_parameter_p = np.abs(np.array(fp["coherence/evolution"][2, :]))        
    axes[2, 2].plot(time, vib_parameter_s, linewidth = 3, alpha = 0.5) 
    axes[2, 2].plot(time, vib_parameter_p, linewidth = 3, alpha = 0.5)
    axes[2, 2].set_xlabel('time (s)', fontsize = 12)
    axes[2, 2].set_ylabel('vibration parameters (a. u.)', fontsize = 12)
    axes[2, 2].yaxis.set_label_coords(-0.26, 0.5)
    axes[2, 2].set_box_aspect(1)
    
    
    vib_parameter_s = np.abs(np.array(fs["coherence/evolution"][3, :]))
    vib_parameter_p = np.abs(np.array(fp["coherence/evolution"][3, :]))        
    axes[2, 3].plot(time, vib_parameter_s, linewidth = 3, alpha = 0.5) 
    axes[2, 3].plot(time, vib_parameter_p, linewidth = 3, alpha = 0.5)
    axes[2, 3].set_xlabel('time (s)', fontsize = 12)
    axes[2, 3].set_ylabel('vibration parameters (a. u.)', fontsize = 12)
    axes[2, 3].yaxis.set_label_coords(-0.26, 0.5)
    axes[2, 3].set_box_aspect(1)
    
    #--------------------------------
    
    position_list = list()
    
    for idx in range(5):
        
        focus_svd_test = screen(optic_file = file_svd_result, n_vector = 1)
        focus_svd_test.n_vector = 200
        focus_svd_test.n = 200
        focus_svd_test.ratio = np.ones(200)
        focus_svd_test.cmode = deepcopy(vibration_cmode[200 * idx : 200 * (idx + 1)])
        
        focus_svd_test.decomposition()
        
        svd_ratio = np.abs(focus_svd_test.ratio[0 : 15])**2 / np.sum(
            np.abs(focus_svd_test.ratio[0 : 50])**2
            )
        
        focus_svd_test.cal_intensity()
        position_list.append([center_index(
            np.sqrt(focus_svd_test.intensity), focus_svd_test.xtick, 
            focus_svd_test.ytick), 
            np.copy(focus_svd_test.intensity), svd_ratio[0]])
        
    for idx in range(5):  
        axes[3, 0].plot(
            np.linspace(-996, 1004, 512), np.sum(position_list[idx][1], 0), 
            linewidth = 3, alpha = 0.5
            )
    axes[3, 0].set_xlabel('x (nm)', fontsize = 12)
    axes[3, 0].set_ylabel('intensity (a. u.)', fontsize = 12)
    axes[3, 0].set_xlim([-250, 250])
    axes[3, 0].set_ylim([-0.3, 4])
    axes[3, 0].yaxis.set_label_coords(-0.26, 0.5)
    axes[3, 0].set_box_aspect(1)
    
    for idx in range(5):  
        axes[3, 1].plot(
            np.linspace(-996, 1004, 512), np.sum(position_list[idx][1], 1), 
            linewidth = 3, alpha = 0.5
            )
    axes[3, 1].set_xlabel('y (nm)', fontsize = 12)
    axes[3, 1].set_ylabel('intensity (a. u.)', fontsize = 12)
    axes[3, 1].set_xlim([-250, 250])
    axes[3, 1].set_ylim([-0.3, 4])
    axes[3, 1].yaxis.set_label_coords(-0.26, 0.5)
    axes[3, 1].set_box_aspect(1)
    
    position_x = np.array([position_list[idx][0][0] for idx in range(5)])
    position_y = np.array([position_list[idx][0][1] for idx in range(5)])
    axes[3, 2].scatter(np.linspace(0, 0.2, 5), position_x * 1e9)
    axes[3, 2].plot(
        np.linspace(0, 0.2, 5), position_x * 1e9, 
        linewidth = 3, alpha = 0.5
        )
    axes[3, 2].scatter(np.linspace(0, 0.2, 5), position_y * 1e9)
    axes[3, 2].plot(
        np.linspace(0, 0.2, 5), position_y * 1e9, 
        linewidth = 3, alpha = 0.5
        )
    
    axes[3, 2].set_xlabel('time (s)', fontsize = 12)
    axes[3, 2].set_ylabel('position (nm)', fontsize = 12)
    axes[3, 2].set_ylim([-50, 50])
    axes[3, 2].yaxis.set_label_coords(-0.26, 0.5)
    axes[3, 2].set_box_aspect(1)
    
    ratio = [position_list[idx][2] for idx in range(5)]
    axes[3, 3].scatter(np.linspace(0, 0.2, 5), ratio)
    axes[3, 3].plot(
        np.linspace(0, 0.2, 5), ratio, linewidth = 3, alpha = 0.5
        )
    axes[3, 3].set_xlabel('time (s)', fontsize = 12)
    axes[3, 3].set_ylabel('coherent ratio', fontsize = 12)
    axes[3, 3].set_ylim([0.75, 1])
    axes[3, 3].yaxis.set_label_coords(-0.26, 0.5)
    axes[3, 3].set_box_aspect(1)   
    
    figure.tight_layout()
    plt.savefig(r"D:\File\Paper\Vibration\codes\sr_source\vibration\figure6.png", dpi = 1000)
    
    

