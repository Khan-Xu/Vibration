# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Thu Oct 10 17:35:43 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: Figure1 and Figure 2
"""

#-----------------------------------------------------------------------------#
# modules

import sys
sys.path.append(r'D:\File\Paper\Vibration\codes\cat')
import os
import time

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from cat.wave_optics.optics import source_optic, screen, ideal_lens
from cat.wave_optics.propagate import fresnel
from cat.wave_optics.widget import plot_optic
from cat.wave_optics._vibration import sinc_vibration

#-----------------------------------------------------------------------------#
# parameters

file_header = r"D:\File\Paper\Vibration\codes\sr_source"
source_file = os.path.join(file_header, r"source\b4_pdr_12400eV.h5")
wavefront_file = os.path.join(file_header, r"source\b4_12400eV_se.npy")

unit = '$\u03bc$m' 

wavefront = np.load(wavefront_file)
wavefront = np.abs(wavefront) * np.exp(-1j * np.angle(wavefront))
vib_mag = [4.3, 8.6, 12.9, 17.2]

#-----------------------------------------------------------------------------#
# functions

def vib_visulation_mode(optic):
    
    figure, axes = plt.subplots(2, 4, figsize = (12, 6))
    
    # plot intensity
    
    for idx in range(4):
        
        normalized_mode = (
            np.abs(optic.cmode[idx])**2 / np.max(np.abs(optic.cmode[idx])**2)
            )
        axes[0, idx].imshow(
            normalized_mode[256 - 102 : 256 + 102, 256 - 102 : 256 + 102], 
            extent = [-200, 200, -200, 200]
            )
        axes[0, idx].set_title("vib-mode @index: %d" % (idx))
        axes[0, idx].set_xlabel('x (%s)' % (unit), fontsize = 12)
        axes[0, idx].set_ylabel('y (%s)' % (unit), fontsize = 12)
        axes[0, idx].yaxis.set_label_coords(-0.26, 0.5)
        
        axes[1, 0].plot(
            np.linspace(-200, 200, 204), normalized_mode[256, 256 - 102 : 256 + 102],
            linewidth = 3, alpha = 0.5
            )
    
    axes[1, 0].set_title("vib_mode @ y = 0 (%s)" % (unit), fontsize = 12)
    axes[1, 0].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[1, 0].set_ylabel('intensity (a. u.)', fontsize = 12)
    axes[1, 0].yaxis.set_label_coords(-0.26, 0.5)
    
    # plot parameters
    
    time_series = np.linspace(0, 10, optic.n - 1)
    ratio = optic.ratio / sum(optic.ratio)
    
    occup = list()
    for idx in range(int(optic.n - 2)):
        value = (
            np.sum(np.array(optic.ratio[0 : int(idx + 1)])**2) /
            np.sum(np.array(optic.ratio)**2)
            )
        occup.append(value)

    for idx in range(3):
        axes[1, 1].plot(time_series, np.abs(optic.evolution[idx, :]), linewidth = 3, alpha = 0.5)
    
    axes[1, 1].set_xlabel('vibration time (ms)', fontsize = 12)
    axes[1, 1].set_ylabel('vibration parameters (a. u.)', fontsize = 12)
    axes[1, 1].yaxis.set_label_coords(-0.26, 0.5)
    axes[1, 1].set_box_aspect(1)
    
    axes[1, 2].scatter(np.arange(16), ratio[0 : 16])
    axes[1, 2].plot(np.arange(16), ratio[0 : 16], linewidth = 3, alpha = 0.5)
    axes[1, 2].set_xlabel("index (n)", fontsize = 12)
    axes[1, 2].set_ylabel("singular value (a. u.)", fontsize = 12)
    axes[1, 2].yaxis.set_label_coords(-0.26, 0.5)
    axes[1, 2].set_box_aspect(1)
    
    axes[1, 3].set_title("")
    axes[1, 3].scatter(np.arange(16), occup[0 : 16])
    axes[1, 3].plot(np.arange(16), occup[0 : 16], linewidth = 3, alpha = 0.5)
    axes[1, 3].set_xlabel("index (n)", fontsize = 12)
    axes[1, 3].set_ylabel("occupation (a. u.)", fontsize = 12)
    axes[1, 3].yaxis.set_label_coords(-0.26, 0.5)
    axes[1, 3].set_box_aspect(1)
    
    figure.tight_layout()

#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    #--------------------------------------------------------------------
    # the construction of source
    
    b4_sr = source_optic(source_file_name = source_file, n_vector = 1, position = 10)
    sr_screen = screen(optic = b4_sr, n_vector = 1, position = 10)
    fresnel(b4_sr, sr_screen)
    
    sr_screen.interp_optic(pixel = [7.8e-6/4, 7.8e-6/4], coor = [1e-3, 1e-3])
    sr_screen.interp_optic(power2 = True)
    sr_screen.cmode[0] = wavefront
    sr_screen.interp_optic(coor = [1e-3, 1e-3])
    
    #--------------------------------------------------------------------
    # the source vibration
    
    # vibration frequency 100 HZ
    
    sr_count = 200
    sr_gif = os.path.join(
        file_header, r"vibration\sr_100hz_vibration_svd_presentation.gif"
        )
    vibration_wavefront = list()
    
    for vibration_position in sinc_vibration(
            [vib_mag[1] * 1e-6], [100], [0], np.linspace(0, 0.01, sr_count)
            ):
        
        sr_100hz = deepcopy(sr_screen)
        sr_100hz.shift(offx = vibration_position)
        vibration_wavefront.append(sr_100hz.cmode[0])
    
    sr_100hz.cmode = vibration_wavefront[:-1]
    sr_100hz.n = sr_count - 1
    sr_100hz.ratio = np.ones(sr_100hz.n)
    
    #--------------------------------------------------------------------
    # plot gif
        
    test_sr_100hz = plot_optic(sr_100hz)
    # test_sr_100hz.export_gif(sr_gif, kind = "intensity", fps = 100)
    
    #--------------------------------------------------------------------
    # decomposition
        
    sr_100hz.decomposition()
    sr_100hz.n = sr_count
    vib_visulation_mode(sr_100hz)
    plt.savefig(r"D:\File\Paper\Vibration\codes\sr_source\vibration\figure1.png", dpi = 1000)
    
    sr_100hz.name = os.path.join(
        file_header, r"vibration\sr_100hz_svd_method_presentation.h5"
        )
    sr_100hz.save_h5()
    
    occup = list()
    for idx in range(int(sr_100hz.n - 2)):
        value = (
            np.sum(np.array(sr_100hz.ratio[0 : int(idx + 1)])**2) /
            np.sum(np.array(sr_100hz.ratio)**2)
            )
        occup.append(value)
        
    #--------------------------------------------------------------------
    # reconstruction
    
    rmse = list()
    wfr_boundary_list = list()
    occupation = list()
    
    origin_wfr = np.abs(vibration_wavefront[0])**2
    origin_wfr = origin_wfr / np.max(origin_wfr)
    
    for index in range(2, 20):
        
        wfr_boundary = (
            sr_100hz.evolution[0, 0] * sr_100hz.ratio[0] * sr_100hz.cmode[0]
            )
        
        for idx in range(index):
            wfr_boundary += (
                sr_100hz.evolution[idx + 1, 0] * sr_100hz.ratio[idx + 1] * 
                sr_100hz.cmode[idx + 1]
                )
        rmse.append(np.sqrt(np.sum(np.abs(
            origin_wfr**0.5 - 
            (np.abs(wfr_boundary)**2 / np.max(np.abs(wfr_boundary)**2))**0.5
            )**2) / 512**2) * 100)
            
        wfr_boundary_list.append(
            np.abs(wfr_boundary)**2 / np.max(np.abs(wfr_boundary)**2)
            )
        occupation.append(occup[index])
    
    #--------------------------------------------------------------------
    # show vibration
    
    figure, axes = plt.subplots(2, 4, figsize = (12, 6))
    
    #------------------------------------------------
    # different vibration modes combination
    
    axes[0, 0].imshow(
        origin_wfr[256 - 102 : 256 + 102, 256 - 102 : 256 + 102], 
        extent = [-200, 200, -200, 200]
        )
    axes[0, 0].set_title("original wavefront")
    axes[0, 0].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[0, 0].set_ylabel('y (%s)' % (unit), fontsize = 12)
    axes[0, 0].yaxis.set_label_coords(-0.26, 0.5)
    
    axes[0, 1].imshow(
        wfr_boundary_list[0][256 - 102 : 256 + 102, 256 - 102 : 256 + 102], 
        extent = [-200, 200, -200, 200]
        )
    axes[0, 1].set_title("2 vib-modes")
    axes[0, 1].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[0, 1].set_ylabel('y (%s)' % (unit), fontsize = 12)
    axes[0, 1].yaxis.set_label_coords(-0.26, 0.5)
    
    axes[0, 2].imshow(
        wfr_boundary_list[2][256 - 102 : 256 + 102, 256 - 102 : 256 + 102], 
        extent = [-200, 200, -200, 200]
        )
    axes[0, 2].set_title("4 vib-modes")
    axes[0, 2].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[0, 2].set_ylabel('y (%s)' % (unit), fontsize = 12)
    axes[0, 2].yaxis.set_label_coords(-0.26, 0.5)
    
    axes[0, 3].imshow(
        wfr_boundary_list[4][256 - 102 : 256 + 102, 256 - 102 : 256 + 102], 
        extent = [-200, 200, -200, 200]
        )
    axes[0, 3].set_title("6 vib-modes")
    axes[0, 3].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[0, 3].set_ylabel('y (%s)' % (unit), fontsize = 12)
    axes[0, 3].yaxis.set_label_coords(-0.26, 0.5)

    #------------------------------------------------
    # different vibration modes combination
    
    axes[1, 0].plot(range(2, 20), rmse, linewidth = 3, alpha = 0.5)
    axes[1, 0].scatter(range(2, 20), rmse)
    axes[1, 0].set_xlabel('index (n)', fontsize = 12)
    axes[1, 0].set_ylabel('rmse (%)', fontsize = 12)
    axes[1, 0].yaxis.set_label_coords(-0.26, 0.5)
    axes[1, 0].set_box_aspect(1)
    
    origin_wfr_line = origin_wfr[256, 256 - 102 : 256 + 102]

    axes[1, 1].plot(
        np.linspace(-200, 200, 204), 
        origin_wfr_line, linewidth = 3, alpha = 0.5
        )
    axes[1, 1].plot(
        np.linspace(-200, 200, 204), 
        wfr_boundary_list[0][256, 256 - 102 : 256 + 102],
        linewidth = 3, alpha = 0.7
        )    
    axes[1, 1].set_title("2 vib_modes @ y = 0 (%s)" % (unit), fontsize = 12)
    axes[1, 1].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[1, 1].set_ylabel('normalized intensity (a. u.)', fontsize = 12)
    axes[1, 1].yaxis.set_label_coords(-0.26, 0.5)
    axes[1, 1].set_box_aspect(1)
    
    axes[1, 2].plot(
        np.linspace(-200, 200, 204), 
        origin_wfr_line, linewidth = 3, alpha = 0.5
        )
    axes[1, 2].plot(
        np.linspace(-200, 200, 204), 
        wfr_boundary_list[2][256, 256 - 102 : 256 + 102], 
        linewidth = 3, alpha = 0.7
        )    
    axes[1, 2].set_title("4 vib_modes @ y = 0 (%s)" % (unit), fontsize = 12)
    axes[1, 2].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[1, 2].set_ylabel('normalized intensity (a. u.)', fontsize = 12)
    axes[1, 2].yaxis.set_label_coords(-0.26, 0.5)
    axes[1, 2].set_box_aspect(1)
    
    axes[1, 3].plot(
        np.linspace(-200, 200, 204), 
        origin_wfr_line, linewidth = 3, alpha = 0.5
        )
    axes[1, 3].plot(
        np.linspace(-200, 200, 204), 
        wfr_boundary_list[4][256, 256 - 102 : 256 + 102], 
        linewidth = 3, alpha = 0.7
        )    
    axes[1, 3].set_title("6 vib_modes @ y = 0 (%s)" % (unit), fontsize = 12)
    axes[1, 3].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[1, 3].set_ylabel('normalized intensity (a. u.)', fontsize = 12)
    axes[1, 3].yaxis.set_label_coords(-0.26, 0.5)
    axes[1, 3].set_box_aspect(1)
    
    figure.tight_layout()
    plt.savefig(r"D:\File\Paper\Vibration\codes\sr_source\vibration\figure2.png", dpi = 1000)

    
    
    
    
    
    
    
    