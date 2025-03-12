'''
Created on Friday 28.02.2025

@author: Jan Niklas Leutloff

import sys
sys.path.append('gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools/')
import peak_finder as pf

This file includes the following functions:
 - getMask
 - mask_peak_env
 - getCornerMean
 - bgr_adjustment
 - advancedPeakFinder_img
'''

import sys
sys.path.append('gpfs/exfel/exp/SPB/202501/p006933/usr/Software/scratch')
import data_helper as dh

from skimage.feature import peak_local_max

import numpy as np
import pandas as pd
import scipy.ndimage as nd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool


#=========================================================================================================================================


#------------------------
# PeakFinder: VERSION 5.0
#------------------------

def mask_peak_env(r0=42, dr=6):
    '''
    dr: thickness of the ring
    r0: radius of the circle if dr==0 or the inner radius of the ring if dr!=0
    (see also: https://confluence.slac.stanford.edu/display/PSDM/Hit+and+Peak+Finding+Algorithms)
    Returns a circular or ring mask
    '''
    r1 = r0 + dr
    d = 2 * r1 +1
    ring_sel = np.ones((d, d), dtype=np.uint16)
    Y, X = np.ogrid[:d, :d]
    dist_from_center = np.sqrt((X - r1)**2 + (Y - r1)**2)
    
    if r1==r0:
        ring_sel[dist_from_center > r0] = 0
    else:
        ring_sel[dist_from_center <= r0] = 0
        ring_sel[dist_from_center > r1] = 0
        
    return ring_sel

def getCornerMean2(cal):
    '''
    Returns the mean value of the four corners of an image.
    '''
    x1 = np.copy(cal[0][256:511+1,768:1023+1])
    x2 = np.copy(cal[3][0:255+1,768:1023+1])
    x3 = np.copy(cal[4][256:511+1,0:255+1])
    x4 = np.copy(cal[7][0:255+1,0:255+1])
    x = (x1 + x2 + x3 + x4)/4
    return x.mean()

def getCornerMean(img, border=50):
    '''
         ------   ------
        |x1    | |    x2|
        |      | |      |
        |      | |      |
        |x3    | |    x4|
         ------   ------
         
    Returns the mean value of the four corners of an image.
    '''
    x1 = img[0][0:128, 0:128]
    x2 = img[7][0:128, 0:128]
    x3 = img[11][0:128, 0:128]
    x4 = img[12][0:128, 0:128]
    x = (x1 + x2 + x3 + x4)/4
    return x.mean()

def bgr_adjustment(bgr):
    '''
    Returns an empirically determined factor for the given background ratio

    ret ^
        |
    1   +                      _--
        |                     /
        +   ---------___     |
        |               \_   |
    0.8 +                 \_/
        |
        =
        |
    0   +-||-----+---+---+---+---+-->
        0       0.6     0.8      1   bgr
    '''
    if bgr>0.95: return 1
    elif bgr>0.85:
        return 1 - 0.2 * np.exp(- (bgr-0.85)**2/(2*0.03**2))
    elif bgr>0.7:
        return 0.9 - 0.1 * np.exp(- (bgr-0.85)**2/(2*0.04**2))
    else:
        return 0.9

def peak_finder_subdet(img, module, threshold, npix_min, rank, inner_mask, outer_mask, r0, dr, snr_min, run, t_id, p_id, printen):

    ret_dict = {'run': run, 't_id': t_id, 'p_id': p_id, 'module': module,'peak_pos_cm': [], 'peak_pos': [],
                'snr': [], 'npix': [], 'integrated': [], 'intensityPerPixel': [], 'peak_env_threshold': []}

                                                                        # To exclude the bright row/colomn
                                            #21                         # at the border of the img array 
    peaks = peak_local_max(img, min_distance=21, threshold_abs=threshold, exclude_border=0)#rank+1

    img_shape=np.shape(img)

    for p in peaks:
        x = p[1]
        y = p[0]
        
        # The numbers on the left side are given by the shape of the given image
        x_min_rank = x-rank if x-rank>0 else 0           #np.max([0, x-rank])
        x_max_rank = x+rank+1 if x+rank+1<img_shape[1] else img_shape[1] #np.min([1024, x+rank+1])
        y_min_rank = y-rank if y-rank>0 else 0           #np.max([0, y-rank])
        y_max_rank = y+rank+1 if y+rank+1<img_shape[0] else img_shape[0] #np.min([512, y+rank+1])
        
        peak_env = np.copy(img[y_min_rank:y_max_rank, x_min_rank:x_max_rank])
        new_peak_env = np.copy(peak_env)

        peak_value = img[y][x]
        
        peak_env_1d = peak_env.reshape(-1)                   # 261 ns ± 9.17 ns

        # Look only at the current peak if there are other more intense peaks in the peak_env
        if peak_env_1d.max()!=peak_value:
            peak_env_1d = peak_env_1d[np.where(peak_env_1d<=peak_value)]

        # sorting it before imgculating the histogram is faster
        sort_1d = np.sort(peak_env_1d)
        sort_1d_size = sort_1d.size-1
        q90 = sort_1d[int(np.round(sort_1d_size*0.9))]
        q10 = sort_1d[int(np.round(sort_1d_size*0.1))]
        median = sort_1d[int(np.round(sort_1d_size*0.5))]
        mean = peak_env_1d.mean()

        # int(number+1) is faster than int(np.ceil(number))      10.5 µs ± 585 ns compared to 103 ns ± 4.97 ns
        # Exclude 0 because there is a maximum which makes it impossible to find the interesting maximum
        bins = np.arange(1, int(peak_value+1)+1)               # 67 µs ± 576 ns
        data, _ = np.histogram(peak_env_1d, bins)              # 3.67 ms ± 37.9 µs
        
        try:
            pos = np.argmax(data) + 1                          # 86.5 µs ± 2.02 µs
            wid = np.argwhere(data[pos:]<data.max()*0.1)[0,0]  # 442 µs ± 6.16 µs

            background_ratio = data[:int(pos+wid)].sum()/data.sum()

            if peak_value-pos<400 and mean-median<15 and background_ratio>0.94: continue

            rel_size = peak_env.size/(2*rank + 1)**2
            peak_env_threshold = int( (pos + (q90-q10)) * bgr_adjustment(background_ratio) * rel_size**(1/4) + 1)
                        
        except IndexError:
            if printen: print('Not choosen:', p)
            continue
        
        new_peak_env[new_peak_env<peak_env_threshold]=0
        
        labeled_peak_env, _ = nd.label(new_peak_env)#np.ones((3,3), dtype=int) 247 µs ± 2.86 µs
        x_in_peak_env = x if x<rank else rank       #np.min([x, rank])
        y_in_peak_env = y if y<rank else rank       #np.min([y, rank])
        peak_label = labeled_peak_env[y_in_peak_env, x_in_peak_env]
        if peak_label==0: 
            if printen: print('Not choosen:', p)
            continue
        npix = np.sum(labeled_peak_env==peak_label)
        
        original_npix_min = npix_min
        if rel_size < 1:
            npix_min = int( npix_min * (rel_size)**(1/6) )
            if printen:
                print('Size of the peak env: {}'.format(rel_size*(2*rank+1)**2))
                print('New npix_min {} for peak {}'.format(npix_min, p))
        
        if npix >= npix_min:
            cm_in_peak_env = np.asarray(np.round(nd.center_of_mass(peak_env, labeled_peak_env, peak_label)), dtype=int)
            
            actual_peak = [cm_in_peak_env[0] + y_min_rank, cm_in_peak_env[1] + x_min_rank]
            
            x = actual_peak[1]
            y = actual_peak[0]
            
            # The numbers on the left side are given by the shape of the given image
            x_min_r0 = x-r0 if x-r0>0 else 0           #np.max([0, x-r0])
            x_max_r0 = x+r0+1 if x+r0+1<img_shape[1] else img_shape[1] #np.min([1024, x+r0+1])
            y_min_r0 = y-r0 if y-r0>0 else 0           #np.max([0, y-r0])
            y_max_r0 = y+r0+1 if y+r0+1<img_shape[0] else img_shape[0] #np.min([512, y+r0+1])

            r0_env = np.copy(img[y_min_r0:y_max_r0, x_min_r0:x_max_r0])
            r0_mask = np.copy(inner_mask[r0-(y-y_min_r0):r0+(y_max_r0-y), r0-(x-x_min_r0):r0+(x_max_r0-x)])

            r1 = r0 + dr
            
            # The numbers on the left side are given by the shape of the given image
            x_min_r1 = x-r1 if x-r1>0 else 0           #np.max([0, x-r1])
            x_max_r1 = x+r1+1 if x+r1+1<img_shape[1] else img_shape[1] #np.min([1024, x+r1+1])
            y_min_r1 = y-r1 if y-r1>0 else 0           #np.max([0, y-r1])
            y_max_r1 = y+r1+1 if y+r1+1<img_shape[0] else img_shape[0] #np.min([512, y+r1+1])
            
            r1_env = np.copy(img[y_min_r1:y_max_r1, x_min_r1:x_max_r1])
            r1_mask = np.copy(outer_mask[r1-(y-y_min_r1):r1+(y_max_r1-y), r1-(x-x_min_r1):r1+(x_max_r1-x)])
            
            r0_env *= r0_mask
            r1_env *= r1_mask
            
            snr = np.mean(r0_env, where=r0_env!=0)/np.mean(r1_env, where=r1_env!=0)
            
            if snr>=snr_min:
                integrated = nd.sum_labels(peak_env, labels=labeled_peak_env, index=peak_label)
                intensityPerPixel = integrated/npix

                ret_dict['peak_pos_cm'].append(np.asarray(actual_peak, dtype=int))
                ret_dict['peak_pos'].append(np.asarray(p, dtype=int))
                ret_dict['snr'].append(np.round(snr, 4))
                ret_dict['npix'].append(npix)
                ret_dict['integrated'].append(int(np.round(integrated, 0)))
                ret_dict['intensityPerPixel'].append(int(np.round(intensityPerPixel, 0)))
                ret_dict['peak_env_threshold'].append(peak_env_threshold)

                if printen: print(actual_peak, snr, npix, integrated, intensityPerPixel, p, peak_env_threshold)
            
        elif printen: print('Not choosen:', p, npix)
            
        npix_min = original_npix_min

    # The DataFrame will be empty if no peak was found although the run and event number was written in ret_dict
    ret_df = pd.DataFrame(ret_dict)
    
    return ret_df

def advancedPeakFinder_img(img, mask, amp=14, npix_min=324, rank=80, r0=42, dr=6, snr_min=1.55, run=-1, t_id=-1, p_id=-1, plot=False, printen=False, debug=0):
    '''
    !!! You should use at least as many CPUs as there are detector modules !!!
    
    Identifies the peaks in the panels of a given imgib-array within the following steps:
    1. Apply peak_loimg_max to image
    2. Create peak_env around found peaks
    3. Set all pixel which are below a threshold to zero
    4. Apply label to peak_env
    5. If the labeled area is larger than npix_min keep the peak
    6. Use center_of_gravity to imgculate the actual peak position
    7. Check snr>snr_min
    
    Parameters
    ----------
    img : numpy.ndarray
        Input data representing a 2d image to be analyzed.
        
    mask : numpy.ndarray
        Same shape of the input array that mask all bad pixels and
        also the area in which no peaks should be searched
        
    amp : float, optional
        Defines the threshold intensity of the peaks. Default is 14.
    
    npix_min : int, optional
        Number of pixels making up the peak. Default is 324.
        
    rank : int, optional
        Defines a region around the peak in which nd.label is used to 
        find the pixels which contribute to the peak.
        Default is 80.
        
    r0 : float, optional
    dr : float, optional
        Parameters r0 and dr evaluate the snr of a peak.
        Default of r0 is 42.
        Default of dr is 6.
        
    snr_min : float, optional
        Minimum snr the peaks have to fulfill. Default is 1.55.

    run, tag : int, optional
        Run and event number under which the data is saved in a pd.DataFrame

    Returns
    -------
    List of peaks.
    Each enty consists of the peak position,
    the snr and npix of the peak.
    '''
    
    masked_img = img * mask
    corner_mean = getCornerMean(img) * amp
    threshold = corner_mean if corner_mean<500 else 500
    threshold += 25
    
    if debug>0: threshold = debug
    if printen: print('{} * {} + 10 = {}'.format(getCornerMean(img), amp, threshold))
    
    inner_mask = mask_peak_env(r0=r0, dr=0)
    outer_mask = mask_peak_env(r0=r0, dr=dr)

    par_list = []
    for module, subdet in enumerate(masked_img):
        par_list.append([subdet, module, threshold, npix_min, rank, inner_mask, outer_mask, r0, dr, snr_min, run, t_id,  p_id, printen])

    with Pool(processes=8, maxtasksperchild=1) as pool:
        total_peaks = pool.starmap(peak_finder_subdet, par_list)

    ret_df = pd.concat(total_peaks, ignore_index=True)
    
    if False:
        fig, ax = plt.subplots(dpi=150)
        cmap = plt.get_cmap('inferno')
        cmap.set_under(color='black')  # Color for values below vmin (black)
        cmap.set_bad(color='black')
        tmp0 = ax.imshow(assembleImage(masked_img), cmap=cmap, norm=LogNorm(vmin=1, vmax=1e3))
        for si, cm in zip(ret_df['module'], ret_df['peak_pos_cm']):
            center_y, center_x = imageCoordinate(si, list(cm))
            circle = Circle(xy=(center_x, center_y), radius=50, ls='-', linewidth=1, edgecolor='b', facecolor='None')
            ax.add_patch(circle)
        divider0 = make_axes_locatable(ax)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(tmp0, cax=cax0)
        plt.show()

    if plot:
        geom = dh.getGeometry()
        fig, ax = plt.subplots(dpi=150)
        cmap = plt.get_cmap('inferno')
        cmap.set_under(color='black')  # Color for values below vmin (black)
        cmap.set_bad(color='black')
        ax = geom.plot_data(masked_img, ax=ax, axis_units='m', cmap=cmap, norm=LogNorm(vmin=1, vmax=1e3))
        fast_scan = np.asarray([el[1] for el in ret_df['peak_pos_cm']])
        slow_scan = np.asarray([el[1] for el in ret_df['peak_pos_cm']])
        module_no = np.asarray(ret_df['module'])
        positions = geom.data_coords_to_positions(module_no, slow_scan, fast_scan)
        for center_y, center_x in zip(positions[:, 1], positions[:, 0]):
            circle = Circle(xy=(center_x, center_y), radius=50, ls='-', linewidth=1, edgecolor='b', facecolor='None')
            ax.add_patch(circle)
        plt.show()
        
    return ret_df

def getPanelCorner(sign):
    '''
    Parameter
    ---------
    sign : int
        Number of the panel in a calib-array
        
    Returns
    -------
    The y- and x-coordinate of the upper left corner of the panel in a 2D image
    '''
    ref_x = 0
    ref_y = 0
    if sign==0:
        ref_x = 135
        ref_y = 0
    elif sign==1:
        ref_x = 685
        ref_y = 0
    elif sign==2:
        ref_x = 1235
        ref_y = 134
    elif sign==3:
        ref_x = 1785
        ref_y = 134
    elif sign==4:
        ref_x = 0
        ref_y = 1039
    elif sign==5:
        ref_x = 550
        ref_y = 1039
    elif sign==6:
        ref_x = 1100
        ref_y = 1173
    elif sign==7:
        ref_x = 1650
        ref_y = 1173

    return ref_y, ref_x

def assembleImage(calib):
    '''
    Parameter
    ---------
    calib : numpy.ndarray
        Input data representing a calib-array that should be used to assemble a 2D image 
        
    Returns
    -------
    The assembled 2D image
    '''
    
    image = np.zeros((2203, 2299))
    
    for sign, subdet in enumerate(calib):
        
        trafo = np.flipud(np.rot90(subdet, axes=(1, 0)))
        
        ref_y, ref_x = getPanelCorner(sign)
        
        for i in range(8):
            col = i%2
            row = i//2
            image[ref_y+row*258:ref_y+256+row*258, ref_x+col*258:ref_x+256+col*258] = trafo[row*256:256+row*256, col*256:256+col*256]
            
    return image

def imageCoordinate(coordinate, sign):
    '''
    Parameter
    ---------
    coordinate : list of [int, int]
        [y-coordinate, x-coordinate]
    sign : int
        Number of the panel in a calib-array
    
        Original:         After the trafo:
        0 | 1 | 2 | 3     7 | 3
        - + - + - + -     - + -
        4 | 5 | 6 | 7     6 | 2
                          - + -
                          5 | 1
                          - + -
                          4 | 0

    Returns
    -------
    New y- and x-coordinate in an assembled image
    '''
    if isinstance(sign, list) and isinstance(coordinate, int):
        tmp = sign
        sign = coordinate
        coordinate = tmp
    
    [y_index, x_index] = coordinate
    
    y_trafo = 1023 - x_index
    x_trafo = 511 - y_index
    
    if x_trafo<=255 and y_trafo<=255:    # 7
        add_x = 0
        add_y = 0
    elif x_trafo<=511 and y_trafo<=255:  # 3
        add_x = 2
        add_y = 0
    elif x_trafo<=255 and y_trafo<=511:  # 6
        add_x = 0
        add_y = 2
    elif x_trafo<=511 and y_trafo<=511:  # 2
        add_x = 2
        add_y = 2
    elif x_trafo<=255 and y_trafo<=767:  # 5
        add_x = 0
        add_y = 4
    elif x_trafo<=511 and y_trafo<=767:  # 1
        add_x = 2
        add_y = 4
    elif x_trafo<=255 and y_trafo<=1023: # 4
        add_x = 0
        add_y = 6
    elif x_trafo<=511 and y_trafo<=1023: # 0
        add_x = 2
        add_y = 6
    
    ref_y, ref_x = getPanelCorner(sign)
    
    new_y = ref_y + add_y + y_trafo
    new_x = ref_x + add_x + x_trafo
    
    return [new_y, new_x]