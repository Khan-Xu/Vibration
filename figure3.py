# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Tue Oct 15 11:23:36 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: Figure 3
"""

#-----------------------------------------------------------------------------#
# modules

import sys
sys.path.append(r'D:\File\Paper\Vibration\codes\cat')
import os

from copy import deepcopy

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import h5py as h5
from scipy import fft

from cat.source import gaussian_schell_source as gss
from cat.wave_optics.optics import source_optic, screen, ideal_lens, kb
from cat.wave_optics.propagate import fresnel, asm, czt, propagate_mode
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

#-----------------------------------------------------------------------------#
# functions

def crl_focus(i_vib):
    
    #---------------------------------------------------
    # the source
    
    b4_sr = source_optic(source_file_name = source_file, n_vector = 1, position = 10)
    sr_screen = screen(optic = b4_sr, n_vector = 1, position = 10)
    fresnel(b4_sr, sr_screen)
    
    sr_screen.interp_optic(pixel = [7.8e-6/4, 7.8e-6/4], coor = [1e-3, 1e-3])
    sr_screen.interp_optic(power2 = True)
    sr_screen.cmode[0] = wavefront
    sr_screen.interp_optic(coor = [1e-3, 1e-3])
    
    #---------------------------------------------------
    # crl
    
    crl = ideal_lens(
        optic = sr_screen, n_vector = 1, i_vector = 0, 
        position = 10, xfocus = 6.667, yfocus = 6.667
        )
    crl.tilt(rotx = i_vib)
    fresnel(sr_screen, crl)
    
    #---------------------------------------------------
    # focus
    
    focus = screen(optic = crl, n_vector = 1, position = 30)
    fresnel(crl, focus)
    
    return focus

def kb_focus(i_vib):
    
    #---------------------------------------------------
    # the source
    
    b4_sr = source_optic(source_file_name = source_file, n_vector = 1, position = 10)
    sr_screen = screen(optic = b4_sr, n_vector = 1, position = 10)
    fresnel(b4_sr, sr_screen)
    
    sr_screen.interp_optic(pixel = [7.8e-6/4, 7.8e-6/4], coor = [1e-3, 1e-3])
    sr_screen.interp_optic(power2 = True)
    sr_screen.cmode[0] = wavefront
    sr_screen.interp_optic(coor = [1e-3, 1e-3])
    
    #---------------------------------------------------
    # crl
    
    vkb = kb(
        optic = sr_screen, direction = 'v', n_vector = 1, 
        position = 10, pfocus = 10, qfocus = 20
        )
    fresnel(sr_screen, vkb)
    
    hkb = kb(
        optic = vkb, direction = 'h', n_vector = 1, 
        position = 10, pfocus = 10, qfocus = 20
        )
    hkb.tilt(rotx = i_vib, kind = 'reflection')
    fresnel(vkb, hkb)

    #---------------------------------------------------
    # focus
    
    focus = screen(optic = hkb, n_vector = 1, position = 30)
    fresnel(hkb, focus)
    
    return focus

#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    #---------------------------------------------------
    # wavefront position vibraiton
    
    b4_sr = source_optic(
        source_file_name = source_file, n_vector = 1, position = 10
        )
    sr_screen = screen(optic = b4_sr, n_vector = 1, position = 10)
    fresnel(b4_sr, sr_screen)
    
    sr_screen.interp_optic(pixel = [7.8e-6/4, 7.8e-6/4], coor = [1e-3, 1e-3])
    sr_screen.interp_optic(power2 = True)
    sr_screen.cmode[0] = wavefront
    sr_screen.interp_optic(coor = [1e-3, 1e-3])
    
    vib_position = [0, 0.5, 10.5]
    vib_wfr = list()
    for i_vib in vib_position:
        sr_screen.shift(offx = i_vib * sr_screen.xpixel)
        vib_wfr.append(sr_screen.cmode[0] / np.abs(sr_screen.cmode[0]).max())
    
    #---------------------------------------------------
    # the mirror angular vibration
    
    vib_angle = 1e-6
    
    vib_focus_crl = [crl_focus(i_vib) for i_vib in [-1e-6, 0, 1e-6]]
    vib_focus_kb = [kb_focus(i_vib) for i_vib in [-1e-6, 0, 1e-6]]
    
    vib_wfr_crl = [
        focus.cmode[0] / np.abs(focus.cmode[0]).max() 
        for focus in vib_focus_crl
        ]
    vib_wfr_kb = [
        focus.cmode[0] / np.abs(focus.cmode[0]).max() 
        for focus in vib_focus_kb
        ]
    
    #---------------------------------------------------
    # plot
    
    figure, axes = plt.subplots(1, 4, figsize = (12, 3))
    
    for idx in range(3):
        axes[0].plot(
            np.linspace(-200, 200, 204), 
            np.abs(vib_wfr[idx][256, 256 - 102 : 256 + 102])**2, linewidth = 3, alpha = 0.5
            )
    axes[0].set_title("position vibration")
    axes[0].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[0].set_ylabel('normalized intensity (a. u.)', fontsize = 12)
    axes[0].yaxis.set_label_coords(-0.26, 0.5)
    axes[0].set_box_aspect(1)
    
    for idx in range(3):
        axes[1].scatter(
            np.linspace(-110, -70, 20), np.abs(vib_wfr[idx][256, 198 : 218])**2
            )
        axes[1].plot(
            np.linspace(-110, -70, 20), np.abs(vib_wfr[idx][256, 198 : 218])**2
            , linewidth = 3, alpha = 0.5
            )
    axes[1].set_title("position vibration")
    axes[1].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[1].set_ylabel('normalized intensity (a. u.)', fontsize = 12)
    axes[1].set_ylim((-0.01, 0.06))
    axes[1].yaxis.set_label_coords(-0.26, 0.5)
    axes[1].set_box_aspect(1)
    
    for idx in range(3):
        axes[2].plot(
            np.linspace(-97, 97, 100), 
            np.abs(vib_wfr_crl[idx][256, 256 - 50 : 256 + 50])**2
            , linewidth = 3, alpha = 0.5
            )
    axes[2].set_title("refactive mirror vibration")
    axes[2].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[2].set_ylabel('normalized intensity (a. u.)', fontsize = 12)
    axes[2].yaxis.set_label_coords(-0.26, 0.5)
    axes[2].set_box_aspect(1)
    
    for idx in range(3):
        axes[3].plot(
            np.linspace(-97, 97, 100), 
            np.abs(vib_wfr_kb[idx][256, 256 - 50 : 256 + 50])**2
            , linewidth = 3, alpha = 0.5
            )
    axes[3].set_title("reflective mirror vibration")
    axes[3].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[3].set_ylabel('normalized intensity (a. u.)', fontsize = 12)
    axes[3].yaxis.set_label_coords(-0.26, 0.5)
    axes[3].set_box_aspect(1)
    
    figure.tight_layout()
    plt.savefig(r"D:\File\Paper\Vibration\codes\sr_source\vibration\figure3.png", dpi = 1000)
    
    

 