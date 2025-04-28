# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------#
# Copyright (c) 
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : Fri May 10 14:59:41 2024"
__email__    = "xuhan@ihep.ac.cn"


"""
Description
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

from scipy import fft

# from cat.source import gaussian_schell_source as gss
from cat.wave_optics.optics import source_optic, screen, ideal_lens, kb, akb
from cat.wave_optics.propagate import fresnel, asm, czt, propagate_mode, propagate_depth
from cat.wave_optics.widget import plot_optic
from cat.wave_optics._vibration import sinc_vibration

#-----------------------------------------------------------------------------#
# parameters

file_header = r"D:\File\Paper\Vibration\sr_source"
source_file = os.path.join(file_header, r"source/b4_pdr_8000eV.h5")
unit = '$\u03bc$m' 

#---------------------------------------------------------
# parameters for vkb

_vkb_ep_location = 79.620 - 0.158/2
_vkb_ep_pfocus = 79.620 - 0.158/2
_vkb_ep_qfocus = 0.76292

_vkb_hb_location = 79.620 + 0.158/2
_vkb_hb_afocus = 0.301
_vkb_hb_bfocus = 0.76292 - 0.158

# parameters for hkb

_hkb_ep_location = 79.870 - 0.079/2
_hkb_ep_pfocus = 79.870 - 0.079/2
_hkb_ep_qfocus = 0.2603

_hkb_hb_location = 79.870 + 0.079/2
_hkb_hb_afocus = 0.0905
_hkb_hb_bfocus = 0.2603 - 0.079

# parameters for vibation

#---------------------------------------------------------
# time and vibration parameters

time = np.linspace(0, 0.2, 1000)

# sr source

# sr

not_sr_vibration = False
flag = 0 if not_sr_vibration else 1

sr_amp_rotx = 3 * [flag * 0.30e-6 * 1.414 / np.sqrt(3)]
sr_phase_rotx = [0.11 * np.pi, -0.33 * np.pi, 0.57 * np.pi]
sr_freq_rotx = [97, 201, 497]

sr_amp_roty = 3 * [flag * 0.12e-6 * 1.414 / np.sqrt(3)]
sr_phase_roty = [0.27 * np.pi, 0.73 * np.pi, -0.91 * np.pi]
sr_freq_roty = [97, 201, 497]

sr_amp_offx = 3 * [flag * 0.88e-6 * 1.414 / np.sqrt(3)]
sr_phase_offx = [0.13 * np.pi, 0.25 * np.pi, 0.55 * np.pi]
sr_freq_offx = [97, 201, 497]

sr_amp_offy = 3 * [flag * 0.23e-6 * 1.414 / np.sqrt(3)]
sr_phase_offy = [0.23 * np.pi, 1.51 * np.pi, 1.71 * np.pi]
sr_freq_offy = [97, 201, 497]

# dcm

not_dcm_vibration = False
flag = 0 if not_dcm_vibration else 1

dcm_amp_rotx = 5 * [flag * 50.0e-09 * 1.414 / np.sqrt(5)]
dcm_phase_rotx = [0.11 * np.pi, -0.33 * np.pi, 0.57 * np.pi, 1.51 * np.pi, 1.71 * np.pi]
dcm_freq_rotx = [10, 33, 57, 83, 107]

# kb

use_akb_vibration = False
flag = 0 if use_akb_vibration else 1

kb_amp_rotx = 3 * [flag * 50.0e-09 * 1.414 / np.sqrt(3)]
kb_phase_rotx = [0.47 * np.pi, 0.11 * np.pi, -0.07 * np.pi]
kb_freq_rotx = [10, 21, 33]

kb_amp_roty = 3 * [flag * 50.0e-09 * 1.414 / np.sqrt(3)]
kb_phase_roty = [0.73 * np.pi, 1.11 * np.pi, -0.26 * np.pi]
kb_freq_roty = [10, 21, 33]

#---------------------------------------------------------
# all the vibration

sr_rotx_list = sinc_vibration(sr_amp_rotx, sr_freq_rotx, sr_phase_rotx, time)
sr_roty_list = sinc_vibration(sr_amp_roty, sr_freq_roty, sr_phase_roty, time)
sr_offx_list = sinc_vibration(sr_amp_offx, sr_freq_offx, sr_phase_offx, time)
sr_offy_list = sinc_vibration(sr_amp_offy, sr_freq_offy, sr_phase_offy, time)

dcm_rotx_list = sinc_vibration(dcm_amp_rotx, dcm_freq_rotx, dcm_phase_rotx, time)
kb_rotx_list = sinc_vibration(kb_amp_rotx, kb_freq_rotx, kb_phase_rotx, time)
kb_roty_list = sinc_vibration(kb_amp_roty, kb_freq_roty, kb_phase_roty, time)

#-----------------------------------------------------------------------------#
# functions

#--------------------------------------------------------------------
# the support function

#--------------------------------------------------------------------
# the beamline function

def beamline_b4_source(time):
    
    #---------------------------------------------------
    # using back propagation to get the source
    
    nanomax_sr = source_optic(
        source_file_name = source_file, n_vector = 1, i_vector = 0, 
        position = 20
        )
    sr_screen = screen(optic = nanomax_sr, n_vector = 1, position = 20)
    fresnel(nanomax_sr, sr_screen)
    sr_screen.cmode[0] = np.abs(sr_screen.cmode[0]) * np.exp(-1j * np.angle(sr_screen.cmode[0]))
    # sr_screen.interp_optic(pixel = [2e-6, 2e-6], coor = [2e-3, 2e-3])
    
    # source vibration
    sr_screen.shift(offx = sr_offx_list[time], offy = sr_offy_list[time])
    sr_screen.tilt(rotx = sr_rotx_list[time], roty = sr_roty_list[time], kind = "source")
    
    return sr_screen

def beamline_b4_100nm_focus_vibration(time, mode = "source"):
    
    #---------------------------------------------------
    # using back propagation to get the source
    
    nanomax_sr = source_optic(source_file_name = source_file, n_vector = 1, i_vector = 0, position = 20)
    sr_screen = screen(optic = nanomax_sr, n_vector = 1, position = 20)
    fresnel(nanomax_sr, sr_screen)
    sr_screen.cmode[0] = np.abs(sr_screen.cmode[0]) * np.exp(-1j * np.angle(sr_screen.cmode[0]))
    
    # source vibration
    sr_screen.shift(offx = sr_offx_list[time], offy = sr_offy_list[time])
    sr_screen.tilt(rotx = sr_rotx_list[time], roty = sr_roty_list[time], kind = "source")
    
    if mode == "source": return sr_screen
    
    sr_screen.interp_optic(pixel = [2e-6, 2e-6], coor = [2e-3, 2e-3])
    
    #---------------------------------------------------
    # from source to the dcm
    
    dcm_screen = screen(optic = sr_screen, n_vector = 1, position = 40)
    fresnel(sr_screen, dcm_screen)
    dcm_screen.interp_optic(pixel = [4e-6, 4e-6], coor = [1.2e-3, 1.2e-3])

    # dcm vibration
    dcm_screen.tilt(rotx = dcm_rotx_list[time], kind = "reflection")

    if mode == "dcm": return dcm_screen

    dcm_screen.interp_optic(pixel = [2e-6, 2e-6], coor = [2e-3, 2e-3])
    
    #---------------------------------------------------
    # from dcm screen to kb
    
    kb_acceptance = screen(optic = dcm_screen, n_vector = 1, position = 79.1)
    fresnel(dcm_screen, kb_acceptance)
    
    kb_acceptance.interp_optic(coor = [4.2e-4, 4.2e-4])
    kb_acceptance.mask(xcoor = [-1.60e-4/2, 1.60e-4/2], ycoor = [-3.98e-4/2, 3.98e-4/2])

    if mode == "kb_acceptance":     
        return kb_acceptance

    kb_acceptance.interp_optic(pixel = [0.14e-6, 0.14e-6], coor = [4.2e-4, 4.2e-4])
    kb_acceptance.mask(xcoor = [-1.60e-4/2, 1.60e-4/2], ycoor = [-3.98e-4/2, 3.98e-4/2])

    vkb_mirror = kb(
        optic = kb_acceptance, direction = 'v', n_vector = 1, 
        position = 79.62, pfocus = 79.620, qfocus = 80 - 79.620 + 0.000247
        )
    fresnel(kb_acceptance, vkb_mirror)
    vkb_mirror.interp_optic(coor = [4.2e-4, 1.8e-4])
    vkb_mirror.tilt(roty = kb_roty_list[time], kind = "reflection")
    # vkb_mirror.interp_optic(coor = [4.2e-4, 1.8e-4])

    if mode == "vkb_mirror": return vkb_mirror
    else:
        vkb_mirror.interp_optic(pixel = [0.14e-6, 0.14e-6], coor = [4.2e-4, 4.2e-4])

    hkb_mirror = kb(
        optic = kb_acceptance, direction = 'h', n_vector = 1, 
        position = 79.840,
        pfocus = 79.840, qfocus = 80 - 79.840
        )
    fresnel(vkb_mirror, hkb_mirror)
    hkb_mirror.tilt(rotx = kb_rotx_list[time], kind = "reflection")
    
    if mode == "hkb_mirror": return hkb_mirror
    
    hkb_mirror.interp_optic(pixel = [0.14e-6, 0.14e-6], coor = [2.0e-4, 2.0e-4])
    hkb_mirror.interp_optic(power2 = True)
    
    focus = screen(optic = nanomax_sr, position = 80, n_vector = 1)
    focus.interp_optic(pixel = [0.35e-8, 0.35e-8], coor = [2.0e-6, 2.0e-6])
    focus.interp_optic(power2 = True)
    
    czt(hkb_mirror, focus)
    
    if mode == "focus": return focus

#-----------------------------------------------------------------------------#
# classes

#-----------------------------------------------------------------------------#
# main

if __name__ == "__main__":
    
    mode = "focus"
    
    #--------------------------------------------------------------------
    # the vibration on propagation

    import time
    vibration_propagation_name = "HEPS_B4_KBfocus_propagation_method_50nrad.h5"
    
    vibration_mode = list()
    
    print("propagation start ... ...", flush = True)

    print(time.asctime( time.localtime(time.time()) ), flush = True)

    for idx in range(1000):

        optic_x = beamline_b4_100nm_focus_vibration(idx, mode = mode)
        vibration_mode.append(optic_x.cmode[0])

        if idx % 100 == 0: 
            print(idx, flush = True)
            print(optic_x.cmode[0].shape, flush = True)
    
    optic_x.cmode = vibration_mode
    optic_x.n = 1000
    optic_x.ratio = np.ones(1000)

    optic_x.decomposition()
    # optic_x.interp_optic(pixel = [0.14e-6, 0.14e-6], coor = [1.8e-4, 1.8e-4])
    optic_x.cmode = optic_x.cmode[0 : 10]
    optic_x.name = vibration_propagation_name
    optic_x.save_h5()

    print(time.asctime( time.localtime(time.time()) ), flush = True)
    
    print("propagation end ... ...", flush = False)

    #--------------------------------------------------------------------
    # the vibration on svd

    #---------------------------------------------------
    # the source
    
    import time
    print(time.asctime( time.localtime(time.time()) ), flush = True)

    print("svd start ... ...", flush = True)

    sr_screen_cmode = list()
    
    for idx in range(1000):
        sr_screen = beamline_b4_source(idx)
        sr_screen_cmode.append(sr_screen.cmode[0])
        
    sr_screen.cmode = sr_screen_cmode
    sr_screen.n = 1000
    sr_screen.ratio = np.ones(1000)
    sr_screen.decomposition()
    sr_screen.n = 10
    
    import time
    print(time.asctime( time.localtime(time.time()) ), flush = True)

    if mode == "source":

        vibration_cmode = list()
        for idx in range(100):
            
            i_optic_mode = np.zeros((sr_screen.xcount, sr_screen.ycount), dtype = np.complex64)
            for i_mirror in range(5):
                i_optic_mode += (
                    sr_screen.evolution[i_mirror, idx] * sr_screen.ratio[i_mirror] * 
                    sr_screen.cmode[i_mirror]
                    )
            
            vibration_cmode.append(i_optic_mode)

        sr_screen0 = deepcopy(sr_screen)
        sr_screen0.cmode = vibration_cmode
        sr_screen0.n = 100
        sr_screen0.ratio = np.ones(100)
        sr_screen0.decomposition()

        sr_screen0.name = "sr_screen_svd_method.h5"
        sr_screen0.save_h5()

    sr_screen.interp_optic(pixel = [2e-6, 2e-6], coor = [2e-3, 2e-3])
    
    print("source finished √", flush = True)

    #---------------------------------------------------
    # the dcm
    
    dcm_screen0 = screen(optic = sr_screen, n_vector = 5, position = 40)
    fresnel(sr_screen, dcm_screen0)
    dcm_screen0.interp_optic(pixel = [4e-6, 4e-6], coor = [1.2e-3, 1.2e-3])
    
    dcm_screen1 = deepcopy(dcm_screen0)
    dcm_screen2 = deepcopy(dcm_screen1)

    dcm_screen_cmode = list()
    
    if mode == "dcm":
        wfr_count = 100
    else:
        wfr_count = 1000

    for idx in range(int(wfr_count)):
        i_mirror_mode = np.zeros((dcm_screen0.xcount, dcm_screen0.ycount), dtype = np.complex64)
        for i_mirror in range(10):
            i_mirror_mode += (
                sr_screen.evolution[i_mirror, idx] * sr_screen.ratio[i_mirror] * 
                dcm_screen0.cmode[i_mirror]
                )
        
        dcm_screen2.cmode[0] = np.copy(i_mirror_mode)
        dcm_screen2.n = 1
        dcm_screen2.tilt(rotx = dcm_rotx_list[idx], kind = "reflection")
        dcm_screen_cmode.append(dcm_screen2.cmode[0])
    
    dcm_screen1.cmode = dcm_screen_cmode
    dcm_screen1.n = int(wfr_count)
    dcm_screen1.ratio = np.ones(int(wfr_count))
    dcm_screen1.decomposition()
    
    print(time.asctime( time.localtime(time.time()) ), flush = True)

    if mode == "dcm":

        dcm_screen1.name = "dcm_screen_svd_method.h5"
        dcm_screen1.save_h5()
    else:
        dcm_screen1.n = 30
        dcm_screen1.n_vector = 30
        dcm_screen1.interp_optic(pixel = [2e-6, 2e-6], coor = [2e-3, 2e-3])
    
    print("dcm finished √", flush = True)

    #---------------------------------------------------
    # the akb slit
    
    kb_acceptance = screen(optic = dcm_screen1, n_vector = 25, position = 79.10)
    fresnel(dcm_screen1, kb_acceptance)
    kb_acceptance.interp_optic(coor = [4.2e-4, 4.2e-4])
    kb_acceptance.mask(
        xcoor = [-1.60e-4/2, 1.60e-4/2], 
        ycoor = [-3.98e-4/2, 3.98e-4/2]
        )
    
    kb_acceptance0 = deepcopy(kb_acceptance)
    
    if mode == "kb_acceptance":
        wfr_count = 100
    else:
        wfr_count = 1000
    
    vibration_cmode = list()
    for idx in range(int(wfr_count)):
        
        i_optic_mode = np.zeros((kb_acceptance.xcount, kb_acceptance.ycount), dtype = np.complex64)
        for i_mirror in range(30):
            i_optic_mode += (
                dcm_screen1.evolution[i_mirror, idx] * dcm_screen1.ratio[i_mirror] * 
                kb_acceptance.cmode[i_mirror]
                )
        vibration_cmode.append(i_optic_mode)
    
    kb_acceptance0.cmode = vibration_cmode
    kb_acceptance0.n = int(wfr_count)
    kb_acceptance0.ratio = np.ones(int(wfr_count))
    kb_acceptance0.decomposition()
    
    kb_acceptance0.n = 5
    kb_acceptance0.n_vector = 5
    
    print(time.asctime( time.localtime(time.time()) ), flush = True)

    if mode == "kb_acceptance":
        kb_acceptance0.name = "kb_acceptance_svd_method.h5"
        kb_acceptance0.save_h5()

    print("kb acceptance finished √", flush = True)

    #---------------------------------------------------
    # the vkb mirror
    
    kb_acceptance0.interp_optic(pixel = [0.14e-6, 0.14e-6], coor = [4.2e-4, 4.2e-4])
    kb_acceptance0.mask(
        xcoor = [-1.60e-4/2, 1.60e-4/2], 
        ycoor = [-3.98e-4/2, 3.98e-4/2]
        )
    time = np.linspace(0, 0.2, 100)
    kb_rotx_list = sinc_vibration(kb_amp_rotx, kb_freq_rotx, kb_phase_rotx, time)
    kb_roty_list = sinc_vibration(kb_amp_roty, kb_freq_roty, kb_phase_roty, time)
    
    vkb_mirror0 = kb(
        optic = kb_acceptance0, direction = 'v', n_vector = 5, 
        position = 79.62, pfocus = 79.620, qfocus = 80 - 79.620 + 0.000247
        )
    fresnel(kb_acceptance0, vkb_mirror0)
    vkb_mirror0.interp_optic(coor = [4.2e-4, 1.8e-4])
    vkb_mirror1 = deepcopy(vkb_mirror0)

    vkb_vibration_all = list()
    
    for i in range(3):
        
        vkb_mirror2 = deepcopy(vkb_mirror0)
        vibration_mode = list()
        
        for idx in range(100):
            vkb_mirror1.cmode[0] = np.copy(vkb_mirror0.cmode[i])
            vkb_mirror1.tilt(roty = kb_roty_list[idx], kind = "reflection")
            vibration_mode.append(vkb_mirror1.cmode[0])
        
        vkb_mirror2.cmode = vibration_mode
        vkb_mirror2.n = 100
        vkb_mirror2.ratio = np.ones(100)
        # vkb_mirror2.interp_optic(coor = [4.2e-4, 1.8e-4])
        
        vkb_mirror2.decomposition()
        vkb_vibration_all.append(deepcopy(vkb_mirror2))

        import time
        print(time.asctime( time.localtime(time.time()) ), flush = True)
    
    print("vkb finished √", flush = True)

    #---------------------------------------------------
    # propagation from vkb to hkb

    hkb_mirror = kb(
        optic = kb_acceptance0, direction = 'h', n_vector = 5, 
        position = 79.840, pfocus = 79.840, qfocus = 80 - 79.840
        )
    hkb_mirror.n = 5
    hkb_vibration_all = list()
    
    for i in range(3):

        hkb_mirror0 = deepcopy(hkb_mirror)
        vkb_mirror0 = deepcopy(vkb_vibration_all[i])

        vkb_mirror0.n = 5
        vkb_mirror0.n_vector = 5
        hkb_mirror0.n = 5
        hkb_mirror0.n_vector = 5
    
        vkb_mirror0.interp_optic(pixel = [0.14e-6, 0.14e-6], coor = [4.2e-4, 4.2e-4])
        fresnel(vkb_mirror0, hkb_mirror0)
        hkb_mirror0.interp_optic(pixel = [0.14e-6, 0.14e-6], coor = [2.0e-4, 2.0e-4])
    
        hkb_vibration_all.append(hkb_mirror0)
    
    import time
    print(time.asctime( time.localtime(time.time()) ), flush = True)
    
    #---------------------------------------------------
    # restore the wavefront
    
    if mode == "focus":
            
        from skimage.restoration import unwrap_phase
        
        e_vkb = list()
        
        for i in range(3):
            
            e_vkb_i = list()

            for idx in range(5):
                
                func_real = sp.interpolate.interp1d(
                    np.linspace(0, 0.2, 100), 
                    np.real(vkb_vibration_all[i].evolution[idx, :]), kind = 'quadratic'
                    )
                func_image = sp.interpolate.interp1d(
                    np.linspace(0, 0.2, 100), 
                    np.imag(vkb_vibration_all[i].evolution[idx, :]), kind = 'zero'
                    )
                e_vkb_i.append(
                    func_real(np.linspace(0, 0.2, 1000)) + 
                    1j * func_image(np.linspace(0, 0.2, 1000))
                    )
            
            e_vkb.append(e_vkb_i)
        
        vibration_cmode = list()

        if mode == "hkb_mirror":
            wfr_count = 10
        else:
            wfr_count = 1000
        
        for idx in range(int(1000)):

            i_sr_mode = np.zeros((hkb_mirror0.xcount, hkb_mirror0.ycount), dtype = np.complex64)
            for i_sr in range(3):
                
                i_mirror_mode = np.zeros(
                    (hkb_mirror0.xcount, hkb_mirror0.ycount), dtype = np.complex64
                    )
                for i_mirror in range(5):
                    i_mirror_mode += (
                        e_vkb[i_sr][i_mirror][idx] * vkb_vibration_all[i_sr].ratio[i_mirror] * 
                        hkb_vibration_all[i_sr].cmode[i_mirror]
                        )
                i_sr_mode += (
                    i_mirror_mode * kb_acceptance0.evolution[i_sr, idx] * 
                    kb_acceptance0.ratio[i_sr]
                    )
            
            vibration_cmode.append(i_sr_mode)

        print(time.asctime( time.localtime(time.time()) ), flush = True)
        
        hkb_mirror = deepcopy(hkb_vibration_all[0])
        hkb_mirror.cmode = vibration_cmode
        hkb_mirror.n = 1000
        hkb_mirror.ratio = np.ones(1000)
    
        hkb_mirror2 = deepcopy(hkb_mirror)

        time = np.linspace(0, 0.2, 1000)
        kb_rotx_list = sinc_vibration(kb_amp_rotx, kb_freq_rotx, kb_phase_rotx, time)

        vibration_cmode = list()
        for i in range(1000):
            hkb_mirror2.cmode = [np.copy(hkb_mirror.cmode[i])]
            hkb_mirror2.n = 1
            hkb_mirror2.n_vector = 1
            hkb_mirror2.tilt(rotx = kb_rotx_list[i], kind = "reflection")
            vibration_cmode.append(np.copy(hkb_mirror2.cmode[0]))

        hkb_mirror_part1 = deepcopy(hkb_vibration_all[0])
        hkb_mirror_part1.cmode = vibration_cmode[0 : 500]
        hkb_mirror_part1.n = 500
        hkb_mirror_part1.ratio = np.ones(500)
        hkb_mirror_part1.decomposition()
        hkb_mirror_part1.cmode = deepcopy(hkb_mirror_part1.cmode[0 : 10])
        hkb_mirror_part1.n = 10
        hkb_mirror_part1.n_vector = 10

        hkb_mirror_part2 = deepcopy(hkb_vibration_all[0])
        hkb_mirror_part2.cmode = vibration_cmode[500 :]
        hkb_mirror_part2.n = 500
        hkb_mirror_part2.ratio = np.ones(500)
        hkb_mirror_part2.decomposition()
        hkb_mirror_part2.cmode = deepcopy(hkb_mirror_part2.cmode[0 : 10])
        hkb_mirror_part2.n = 10
        hkb_mirror_part2.n_vector = 10

        import time
        print(time.asctime( time.localtime(time.time()) ), flush = True)

        print("hkb finished √", flush = True)

    #---------------------------------------------------
    # from hkb to focus

    hkb_mirror_part1.interp_optic(power2 = True)
    hkb_mirror_part2.interp_optic(power2 = True)
    # hkb_mirror.tilt(rotx = kb_rotx_list[time], kind = "reflection")
    #------------------------------------------------
    # from kb to focus
    
    # if mode == "focus":
        
    focus_part1 = screen(optic = hkb_mirror_part1, position = 80, n_vector = 10)
    focus_part1.n = 10
    focus_part1.interp_optic(pixel = [0.39e-8, 0.39e-8], coor = [2.0e-6, 2.0e-6])
    focus_part1.interp_optic(power2 = True)
    czt(hkb_mirror_part1, focus_part1)

    focus_part2 = screen(optic = hkb_mirror_part2, position = 80, n_vector = 10)
    focus_part2.n = 10
    focus_part2.interp_optic(pixel = [0.39e-8, 0.39e-8], coor = [2.0e-6, 2.0e-6])
    focus_part2.interp_optic(power2 = True)
    czt(hkb_mirror_part2, focus_part2)

    print("focus start ...", flush = True)

    vibration_cmode = list()
    for idx in range(int(500)):
        
        i_optic_mode = np.zeros((focus_part1.xcount, focus_part1.ycount), dtype = np.complex64)
        for i_mirror in range(10):
            i_optic_mode += (
                hkb_mirror_part1.evolution[i_mirror, idx] * hkb_mirror_part1.ratio[i_mirror] * 
                focus_part1.cmode[i_mirror]
                )
        vibration_cmode.append(i_optic_mode)
                
    for idx in range(int(500)):
            
        i_optic_mode = np.zeros((focus_part2.xcount, focus_part2.ycount), dtype = np.complex64)
        for i_mirror in range(10):
            i_optic_mode += (
                hkb_mirror_part2.evolution[i_mirror, idx] * hkb_mirror_part2.ratio[i_mirror] * 
                focus_part2.cmode[i_mirror]
                )
        vibration_cmode.append(i_optic_mode)
        
    focus0 = deepcopy(focus_part1)
    focus0.cmode = vibration_cmode
    focus0.n = 1000
    focus0.n_vector = 1000
    focus0.ratio = np.ones(1000)
    # focus0.name = "focus_svd_wfrs_50urad.h5"
    # focus0.save_h5()
    
    focus0.decomposition()
    focus0.cmode = focus0.cmode[0 : 10]
    focus0.name = "HEPS_B4_KBfocus_svd_method_50nrad.h5"
    focus0.save_h5()

    print("focus finished √", flush = True)
    print(time.asctime( time.localtime(time.time()) ), flush = True)





