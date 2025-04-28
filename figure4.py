# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Wed Oct 16 08:57:56 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description: Figure 4
"""

#-----------------------------------------------------------------------------#
# modules

import sys
sys.path.append(r'D:\File\Paper\Vibration\codes\cat')
import os
import time

from copy import deepcopy

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import h5py as h5
from scipy import fft

from cat.wave_optics.optics import source_optic, screen, ideal_lens
from cat.wave_optics._vibration import sinc_vibration
from cat.wave_optics.widget import gaussian_fit
from cat.wave_optics.propagate import fresnel
from cat.wave_optics.widget import plot_optic

#-----------------------------------------------------------------------------#
# parameters

file_header = r"D:\File\Paper\Vibration\codes\sr_source"
source_file = os.path.join(file_header, r"source\b4_pdr_12400eV.h5")
wavefront_file = os.path.join(file_header, r"source\b4_12400eV_se.npy")

unit = '$\u03bc$m' 
mode_label0 = '$\u03C6$\u2080'

wavefront = np.load(wavefront_file)
wavefront = np.abs(wavefront) * np.exp(-1j * np.angle(wavefront))

#-----------------------------------------------------------------------------#
# functions

def beamline_func(sr_vibration, mirror_vibration, time):
    
    # construct source optic
    
    nanomax_sr = source_optic(source_file_name = source_file, n_vector = 1, position = 10)
    sr_screen = screen(optic = nanomax_sr, n_vector = 1, position = 10)
    fresnel(nanomax_sr, sr_screen)
    
    # load single electron wavefront
    
    sr_screen.interp_optic(pixel = [7.8e-6/4, 7.8e-6/4], coor = [1e-3, 1e-3])
    sr_screen.interp_optic(power2 = True)
    sr_screen.cmode[0] = wavefront
    sr_screen.interp_optic(coor = [1e-3, 1e-3])
    
    sr_screen.shift(offx = sr_vibration[time])
    
    # to the mirror
    
    mirror = ideal_lens(optic = sr_screen, position = 15, xfocus = 10, yfocus = 10)
    fresnel(sr_screen, mirror)
    mirror.shift(offy = mirror_vibration[time])
    
    # to the focus
    
    focus = screen(optic = mirror, n_vector = 1, position = 45)
    fresnel(mirror, focus)
    
    return focus.cmode[0]

def center_index(wavefront, x_range, y_range):
    
    x_para, fitted = gaussian_fit(x_range, np.sum(np.abs(wavefront)**2, 0))
    y_para, fitted = gaussian_fit(y_range, np.sum(np.abs(wavefront)**2, 1))
    
    return [x_para[1], y_para[1]]
    
#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    
    t_start = time.time()
    
    #--------------------------------------------------------------------
    # the construction of source
    
    # construct source optic
    
    test_sr = source_optic(source_file_name = source_file, n_vector = 1, position = 10)
    sr_screen = screen(optic = test_sr, n_vector = 1, position = 10)
    fresnel(test_sr, sr_screen)
    
    # load single electron wavefront
    
    sr_screen.interp_optic(pixel = [7.8e-6/4, 7.8e-6/4], coor = [1e-3, 1e-3])
    sr_screen.interp_optic(power2 = True)
    sr_screen.cmode[0] = wavefront
    sr_screen.interp_optic(coor = [1e-3, 1e-3])
    
    #---------------------------------------------------
    # the source vibration
    
    # vibration frequency 100 HZ
    
    sr_count = 50
    vibration_wavefront = list()
    
    for vibration_position in sinc_vibration([5.0e-6], [100], [0], np.linspace(0, 0.01, sr_count)):
        
        sr_100hz = deepcopy(sr_screen)
        sr_100hz.shift(offx = vibration_position)
        vibration_wavefront.append(sr_100hz.cmode[0])
    
    sr_100hz.cmode = vibration_wavefront
    sr_100hz.n = sr_count
    sr_100hz.ratio = np.ones(sr_100hz.n)
    
    # decomposition
        
    sr_100hz.decomposition()
    sr_100hz.n = 5
    
    #--------------------------------------------------------------------
    # the mirror
    
    sr_100hz.n = 5
    mirror = ideal_lens(optic = sr_100hz, position = 15, xfocus = 10, yfocus = 10)
    fresnel(sr_100hz, mirror)
    
    #---------------------------------------------------
    # the vibration of mirror
    
    # vibration frequency 10 HZ
    
    mirror_count = 50
    h5_path = os.path.join(file_header, r"vibration\mirror_10hz_vibration_@mode%02d.h5")
    
    for idx in range(5):
        
        vibration_wavefront = list()

        for vibration_position in sinc_vibration([5.0e-6], [10], [0], np.linspace(0, 0.1, mirror_count)):
            
            mirror_10hz = deepcopy(mirror)
            mirror_10hz.cmode = [mirror.cmode[idx]]
            mirror_10hz.shift(offy = vibration_position)
            vibration_wavefront.append(mirror_10hz.cmode[0])
        
        mirror_10hz.cmode = vibration_wavefront
        mirror_10hz.ratio = np.ones(mirror_count)

        # decomposition
        
        mirror_10hz.n = mirror_count
        mirror_10hz.decomposition()
        mirror_10hz.name = h5_path % (idx)
        mirror_10hz.save_h5()
    
    #--------------------------------------------------------------------
    # the combination
    
    #---------------------------------------------------
    # interpolate slow vibration, repeat the quick vibration
    
    from skimage.restoration import unwrap_phase
    
    e_mirror = list()
    count = 500
    time_interp = np.linspace(0, 0.1, mirror_count)
    
    mirror_list = [
        screen(optic_file = h5_path % (idx), n_vector = 6, position = 15) for idx in range(5)
        ]
    
    for mirror_screen_i in mirror_list:
        
        e_mirror_i = list()
        
        for idx in range(5):
            
            func_real = sp.interpolate.interp1d(
                np.linspace(0, 0.1, mirror_count), 
                np.real(mirror_screen_i.evolution[idx, :]), kind = "quadratic"
                )
            func_imag = sp.interpolate.interp1d(
                np.linspace(0, 0.1, mirror_count), 
                np.imag(mirror_screen_i.evolution[idx, :]), 
                kind = "quadratic"
                )
            e_mirror_i.append(
                func_real(np.linspace(0, 0.1, count)) + 
                1j * func_imag(np.linspace(0, 0.1, count))
                )
            
        e_mirror.append(np.array(e_mirror_i))
        
    e_sr = np.tile(sr_100hz.evolution[0 : 6, :], 10)
    
    #---------------------------------------------------
    # the combination of the vibraiton modes
    
    vibration_cmode = list()
    
    for idx in range(count):
        
        i_sr_mode = np.zeros((sr_100hz.xcount, sr_100hz.ycount), dtype = complex)
        for i_sr in range(5):
            
            i_mirror_mode = np.zeros((sr_100hz.xcount, sr_100hz.ycount), dtype = complex)
            for i_mirror in range(5):
                i_mirror_mode += (
                    e_mirror[i_sr][i_mirror, idx] * 
                    mirror_list[i_sr].ratio[i_mirror] * 
                    mirror_list[i_sr].cmode[i_mirror]
                    )
            i_sr_mode += i_mirror_mode * e_sr[i_sr, idx] * sr_100hz.ratio[i_sr]
            
        vibration_cmode.append(i_sr_mode)
    
    mirror_screen = screen(optic = mirror_10hz, n_vector = count, position = 15)
    mirror_screen.n = 500
    mirror_screen.cmode = vibration_cmode
    mirror_screen.ratio = np.ones(count)
    mirror_screen.decomposition()
    mirror_screen.n = 20
    
    #--------------------------------------------------------------------
    # the focus
    
    mirror_screen.n = 15
    focus_vib = screen(optic = mirror_screen, position = 45)
    fresnel(mirror_screen, focus_vib)
    
    # reconstruct focus wavefront
    
    vibration_cmode = list()
    
    for idx in range(count):
        i_wfr = np.zeros((focus_vib.xcount, focus_vib.ycount), dtype = complex)
        
        for i_focus in range(15):
            i_wfr += (
                mirror_screen.evolution[i_focus, idx] * 
                mirror_screen.ratio[i_focus] * focus_vib.cmode[i_focus]
                )
        vibration_cmode.append(i_wfr)
        
    focus_vib.n = 500
    focus_vib.cmode = vibration_cmode
    focus_vib.ratio = np.ones(count)
    
    t_end = time.time()
    print("The vib mode method time cost: %.2f s" % (t_end - t_start))
    


    
    #--------------------------------------------------------------------
    # the whole vibration propagation
    
    t_start = time.time()
    
    sr_vibration_position = sinc_vibration([5e-6], [100], [0], np.linspace(0, 0.1, 500))
    mirror_vibration_position = sinc_vibration([5e-6], [10], [0], np.linspace(0, 0.1, 500))
    vibration_cmode_propagate = list()
    
    for idx in range(500):
        i_cmode = beamline_func(
            sr_vibration_position, mirror_vibration_position, idx
            )
        vibration_cmode_propagate.append(i_cmode)
    
    focus_tro = screen(optic = mirror_10hz, n_vector = count, position = 45)
    focus_tro.cmode = vibration_cmode_propagate
    focus_tro.ratio = np.ones(count)
    focus_tro.n = 500
    
    t_end = time.time()
    print("The propagation method time cost: %.2f s" % (t_end - t_start))
    
    
    
    
    #--------------------------------------------------------------------
    # single electron wavefront source size
    
    test_sr = source_optic(source_file_name = source_file, n_vector = 1, position = 10)
    sr_screen = screen(optic = test_sr, n_vector = 1, position = 10)
    fresnel(test_sr, sr_screen)
    
    # load single electron wavefront
    
    sr_screen.interp_optic(pixel = [7.8e-6/4, 7.8e-6/4], coor = [1e-3, 1e-3])
    sr_screen.interp_optic(power2 = True)
    sr_screen.cmode[0] = wavefront
    sr_screen.interp_optic(coor = [1e-3, 1e-3])

    se_sr = screen(optic = sr_screen, n_vector = 1, position = 0)
    fresnel(sr_screen, se_sr)
    
    line = (
        np.sum(np.abs(se_sr.cmode[0])**2, 0) / 
        np.sum(np.abs(se_sr.cmode[0])**2, 0).max()
        )
    para, curve = gaussian_fit(np.linspace(-500, 500, 512), line)
    se_beamsize = para[2] * 1e-6
    
    #--------------------------------------------------------------------
    #plot
    
    
    figure, axes = plt.subplots(3, 4, figsize = (12, 9))
    
    #------------------------------------------------
    # different vibration modes
    
    mirror_result = screen(
        optic_file = h5_path % (0), n_vector = 4, position = 15
        ) 
    xr = np.linspace(-500, 500, 512)
    yr = np.linspace(-500, 500, 512)
    
    for idx in range(2):
        
        axes[0, idx].imshow(
            np.abs(sr_100hz.cmode[idx])**2, extent = [-500, 500, -500, 500]
            )
        axes[0, idx].set_xlabel('x (%s)' % (unit), fontsize = 12)
        axes[0, idx].set_ylabel('y (%s)' % (unit), fontsize = 12)
        axes[0, idx].yaxis.set_label_coords(-0.26, 0.5)
        
        axes[0, idx + 2].imshow(
            np.abs(mirror_result.cmode[idx])**2, 
            extent = [-500, 500, -500, 500]
            )
        axes[0, idx + 2].set_xlabel('x (%s)' % (unit), fontsize = 12)
        axes[0, idx + 2].set_ylabel('y (%s)' % (unit), fontsize = 12)
        axes[0, idx + 2].yaxis.set_label_coords(-0.26, 0.5)
     
    focus_vib_test = deepcopy(focus_vib)
    focus_vib_test.decomposition()
    vib_ratio = np.abs(focus_vib_test.ratio[0 : 15])**2 / np.sum(
        np.abs(focus_vib_test.ratio[0 : 50])**2
        )
    focus_tro_test = deepcopy(focus_tro)
    focus_tro_test.decomposition()
    tro_ratio = np.abs(focus_tro_test.ratio[0 : 15])**2 / np.sum(
        np.abs(focus_tro_test.ratio[0 : 50])**2
        )
    
    for idx in range(4):
        
        axes[1, idx].imshow(
            np.abs(focus_vib_test.cmode[idx][256 - 51 : 256 + 51, 256 - 51 : 256 + 51])**2, 
            extent = [-100, 100, -100, 100]
            )
        axes[1, idx].set_xlabel('x (%s)' % (unit), fontsize = 12)
        axes[1, idx].set_ylabel('y (%s)' % (unit), fontsize = 12)
        axes[1, idx].yaxis.set_label_coords(-0.26, 0.5)
        
    #------------------------------------------------
    # comparation
    
    vib_list, tro_list = list(), list()
    
    for idx in range(count):
        
        vibx_idx, viby_idx = center_index(
            focus_vib.cmode[idx], 
            np.linspace(-500, 500, 512), np.linspace(-500, 500, 512)
            )
        vib_list.append([vibx_idx, viby_idx])
        
        trox_idx, troy_idx = center_index(
            focus_tro.cmode[idx], 
            np.linspace(-500, 500, 512), np.linspace(-500, 500, 512)
            )
        tro_list.append([trox_idx, troy_idx])
    
    axes[2, 0].plot(
        np.array(vib_list)[:, 0], np.array(vib_list)[:, 1], 
        linewidth = 3, alpha = 0.5
        )
    axes[2, 0].plot(
        np.array(tro_list)[:, 0], np.array(tro_list)[:, 1], 
        linewidth = 3, alpha = 0.5
        )
    axes[2, 0].set_xlabel('x (%s)' % (unit), fontsize = 12)
    axes[2, 0].set_ylabel('y (%s)' % (unit), fontsize = 12)
    axes[2, 0].yaxis.set_label_coords(-0.26, 0.5)
    axes[2, 0].set_box_aspect(1)
    
    axes[2, 1].plot(
        np.linspace(0, 0.1, 500)*1e3, np.abs(focus_vib_test.evolution[0, :]), 
        linewidth = 3, alpha = 0.5
        )
    axes[2, 1].plot(
        np.linspace(0, 0.1, 500)*1e3, np.abs(focus_tro_test.evolution[0, :]), 
        linewidth = 3, alpha = 0.5
        )
    axes[2, 1].set_xlabel('vibration time (ms)', fontsize = 12)
    axes[2, 1].set_ylabel('vibration parameters (a. u.)', fontsize = 12)
    axes[2, 1].yaxis.set_label_coords(-0.26, 0.5)
    axes[2, 1].set_box_aspect(1)
        
    #------------------------------------------------
    # coherence ratio
    
    axes[2, 2].scatter(range(15), vib_ratio, alpha = 0.5)
    axes[2, 2].plot(range(15), vib_ratio, linewidth = 3, alpha = 0.5)
    axes[2, 2].scatter(range(15), tro_ratio, linewidth = 3, alpha = 0.5)
    axes[2, 2].plot(range(15), tro_ratio, linewidth = 3, alpha = 0.5)
    
    axes[2, 2].set_xlabel('index (n)', fontsize = 12)
    axes[2, 2].set_ylabel('coherence ratio', fontsize = 12)
    axes[2, 2].yaxis.set_label_coords(-0.26, 0.5)
    axes[2, 2].set_box_aspect(1)
        
    #------------------------------------------------
    # csd
    
    # bessel function

    extended_vib_source = np.sqrt(5e-6**2 + se_beamsize**2)
    
    xr = np.linspace(-500, 500, 512)
    yr = np.linspace(-500, 500, 512)
    
    mirror_screen.cal_csd(direction = 'y')
    bessel_viby = sp.special.j0(xr * 1e-6 * (2 * np.pi * extended_vib_source / (1e-10 * 15.0)))
    axes[2, 3].scatter(
        xr[256 : 256 + 80], mirror_screen.sdc1y[256 : 256 + 80], 
        s = 12, alpha = 0.5
        )
    axes[2, 3].plot(
        xr[256 : 256 + 80], np.abs(bessel_viby[256 : 256 + 80]), 
        linewidth = 3, alpha = 0.5
        )
    
    mirror_screen.cal_csd(direction = 'x')
    bessel_vibx = sp.special.j0(yr * 1e-6 * (2 * np.pi * (extended_vib_source * (2 + 1)) / (1e-10 * 30.0)))
    axes[2, 3].scatter(
        xr[256 : 256 + 80], mirror_screen.sdc1x[256 : 256 + 80], 
        s = 12, alpha = 0.5
        )
    axes[2, 3].plot(
        xr[256 : 256 + 80], np.abs(bessel_vibx[256 : 256 + 80]), 
        linewidth = 3, alpha = 0.5
        )
    
    axes[2, 3].set_xlabel('x or y (%s)' % (unit), fontsize = 12)
    axes[2, 3].set_ylabel('degree of coherence', fontsize = 12)
    axes[2, 3].yaxis.set_label_coords(-0.26, 0.5)
    axes[2, 3].set_box_aspect(1)
    
    figure.tight_layout()
    plt.savefig(r"D:\File\Paper\Vibration\codes\sr_source\vibration\figure4.png", dpi = 1000)
    
            
    