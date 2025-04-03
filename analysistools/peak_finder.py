'''
Created on Thursday 03.04.2025

@author: Jan Niklas Leutloff

import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools')
import peak_finder as pf

This file includes the following functions:
 - getMask
 - mask_peak_env
 - getCornerMean
 - bgr_adjustment
 - advancedPeakFinder_img
'''

import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools')
import data_helper as dh

from skimage.feature import peak_local_max

import numpy as np
import pandas as pd
import scipy.ndimage as nd
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool

cmap = plt.get_cmap('inferno')
cmap.set_under(color='black')  # Color for values below vmin (black)
cmap.set_bad(color='black')

#------------------------
# PeakFinder: VERSION 2.0
#------------------------

def mask_peak_env(r0=30, dr=6):
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


def peak_finder_subdet(img, module, threshold, npix_min, rank, inner_mask, outer_mask, r0, dr, snr_min, run, t_id, p_id, mini_verbose, verbose):
    ret_dict = {'run': run, 'trainId': t_id, 'pulseId': p_id, 'module': module, 'peak_pos_cm': [], 'peak_pos': [],
                'snr': [], 'npix': [], 'integrated': [], 'intensityPerPixel': [], 'peak_env_threshold': []}

                                                                        # To exclude the bright row/colomn
                                            #21                         # at the border of the img array 
    peaks = peak_local_max(img, min_distance=15, threshold_abs=threshold, exclude_border=1)#rank+1
    if verbose: print('Number of peaks in module {}: {}'.format(module, len(peaks)))

    img_shape = img.shape
    img_mean = img.mean()

    for p in peaks:
        x = p[1]
        y = p[0]
        
        # The numbers on the left side are given by the shape of the given image
        x_min_rank = x-rank if x-rank>0 else 0           #np.max([0, x-rank])
        x_max_rank = x+rank+1 if x+rank+1<img_shape[1] else img_shape[1] #np.min([1024, x+rank+1])
        y_min_rank = y-rank if y-rank>0 else 0           #np.max([0, y-rank])
        y_max_rank = y+rank+1 if y+rank+1<img_shape[0] else img_shape[0] #np.min([512, y+rank+1])
        
        peak_env = np.asarray(img[y_min_rank:y_max_rank, x_min_rank:x_max_rank], dtype=float)
        new_peak_env = np.copy(peak_env)

        peak_value = img[y][x]
        
        peak_env_1d = peak_env.reshape(-1)                   # 261 ns ± 9.17 ns

        # Look only at the current peak if there are other more intense peaks in the peak_env
        if peak_env_1d.max()!=peak_value:
            peak_env_1d = peak_env_1d[np.where(peak_env_1d<=peak_value)]

        mean = peak_env_1d.mean()

        # int(number+1) is faster than int(np.ceil(number))      10.5 µs ± 585 ns compared to 103 ns ± 4.97 ns
        rel_size = peak_env.size/(2*rank + 1)**2
        peak_env_threshold = int( mean + 1)

        peak_env_filtered = nd.gaussian_filter(new_peak_env, sigma=0.5, mode='nearest')
        peak_env_filtered[peak_env_filtered<5e-2]=0
        peak_env_filtered[(peak_env_filtered>0) & (peak_env_filtered<1)]=1
        peak_env_mask = new_peak_env<1
        new_peak_env += peak_env_filtered*peak_env_mask.astype(int)
        
        new_peak_env[new_peak_env<peak_env_threshold]=0
        
        labeled_peak_env, _ = nd.label(new_peak_env, structure=getStructure())      # 247 µs ± 2.86 µs
        x_in_peak_env = x if x<rank else rank       #np.min([x, rank])
        y_in_peak_env = y if y<rank else rank       #np.min([y, rank])
        peak_label = labeled_peak_env[y_in_peak_env, x_in_peak_env]
        if peak_label==0: 
            if verbose: print('Not choosen:', p, module)
            continue
        npix = np.sum((labeled_peak_env==peak_label) & (peak_env>=peak_env_threshold))
        
        original_npix_min = npix_min
        if rel_size < 1:
            npix_min = int( npix_min * (rel_size)**(1/6) )
            if verbose:
                print('Size of the peak env: {}, new npix_min {} for peak {}'.format(int(rel_size*(2*rank+1)**2), npix_min, p))
        
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
                ret_dict['intensityPerPixel'].append(np.round(intensityPerPixel, 2))
                ret_dict['peak_env_threshold'].append(peak_env_threshold)

                if verbose | mini_verbose:
                    print(actual_peak, np.round(snr, 4), npix, int(np.round(integrated, 0)), np.round(intensityPerPixel, 2), p, peak_env_threshold)
                    print(npix_min, rel_size)
            
        elif verbose: print('Not choosen:', p, module, npix)
            
        npix_min = original_npix_min

    # The DataFrame will be empty if no peak was found although the run and event number was written in ret_dict
    ret_df = pd.DataFrame(ret_dict)
    
    return ret_df

def advancedPeakFinder_img(img, mask, npix_min=64, rank=50, r0=30, dr=6, snr_min=1.05, run=-1, t_id=-1, p_id=-1, 
                           plot=False, mini_verbose=False, verbose=False, debug=0):
    '''
    !!! You should use at least as many CPUs as there are detector modules !!!
    
    Identifies the peaks in the panels of a given imgib-array within the following steps:
    1. Apply peak_local_max to image
    2. Create peak_env around found peaks
    3. Set all pixel which are below a threshold to zero
    4. Apply label to peak_env
    5. If the labeled area is larger than npix_min keep the peak
    6. Use center_of_gravity to calculate the actual peak position
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
    threshold = np.ceil(img.mean())+0.1
    
    if debug>0: threshold = debug
    if verbose | mini_verbose: print('Threshold for peak_local_max: {}'.format(threshold))

    inner_mask = mask_peak_env(r0=r0, dr=0)
    outer_mask = mask_peak_env(r0=r0, dr=dr)

    par_list = []
    for module, subdet in enumerate(masked_img):
        par_list.append([subdet, module, threshold, npix_min, rank, inner_mask, outer_mask, r0, dr, snr_min, run, t_id,  p_id, mini_verbose, verbose])

    if False:
        with Pool(processes=16, maxtasksperchild=1) as pool:
            total_peaks = pool.starmap(peak_finder_subdet, par_list)
    else:
        total_peaks = []
        for parameter in par_list:
            tmp = peak_finder_subdet(*parameter)
            total_peaks.append(tmp)
    ret_df = pd.concat(total_peaks, ignore_index=True)
    
    if plot:
        fig, ax = plt.subplots(dpi=150)
        cmap = plt.get_cmap('inferno')
        cmap.set_under(color='black')  # Color for values below vmin (black)
        cmap.set_bad(color='black')
        tmp0 = ax.imshow(dh.assembleImage(masked_img), cmap=cmap, norm=LogNorm(vmin=1e-2, vmax=1e2))
        for m_nr, cm in zip(ret_df['module'], ret_df['peak_pos_cm']):
            center_y, center_x = dh.imageCoordinate(list(cm), m_nr)
            circle = Circle(xy=(center_x, center_y), radius=30, ls='-', linewidth=1, edgecolor='b', facecolor='None')
            ax.add_patch(circle)
        divider0 = make_axes_locatable(ax)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(tmp0, cax=cax0)
        plt.show()

    if False:
        geom = dh.getGeometry(run)
        fig, ax = plt.subplots(dpi=150)
        cmap = plt.get_cmap('inferno')
        cmap.set_under(color='black')  # Color for values below vmin (black)
        cmap.set_bad(color='black')
        ax = geom.plot_data(masked_img, ax=ax, axis_units='m', cmap=cmap, norm=LogNorm(vmin=1e-2, vmax=1e1))
        fast_scan = np.asarray([el[1] for el in ret_df['peak_pos_cm']])
        slow_scan = np.asarray([el[0] for el in ret_df['peak_pos_cm']])
        print(fast_scan, slow_scan)
        module_no = np.asarray(ret_df['module'])
        positions = geom.data_coords_to_positions(module_no, slow_scan, fast_scan)
        for center_y, center_x in zip(positions[:, 1], positions[:, 0]):
            circle = Circle(xy=(center_x, center_y), radius=0.005, ls='--', linewidth=1, edgecolor='b', facecolor='None')
            ax.add_patch(circle)
        plt.show()
        
    return ret_df

def getStructure(struc: str='default'):
    if struc=='edges': return np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=int)
    #elif struc=='karo': return np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]], dtype=int)
    #elif struc=='large': return np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]], dtype=int)
    else:
        return np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)

def test_peak_env(img, mask, peak, m_nr, rank=50, npix_min=10):
    #peak, m_nr = [508, 18],4 #dh.stackCoordinate([695, 917])#[166, 27],2 #
    image = img*mask
    total_mean = image.mean()
    total_std = image.std()
    total_median = np.quantile(image, 0.5)
    total_q10 = np.quantile(image, 0.1)
    total_q90 = np.quantile(image, 0.9)
    
    print('Total image ... mean: {}, std: {}, median: {}, q10: {}, q90: {}'.format(total_mean, total_std, total_median, total_q10, total_q90))
    
    img = image[m_nr]
    img_shape=np.shape(img)
    
    x = peak[1]
    y = peak[0]
    print(y, x)
    plt.figure()
    plt.imshow(img, cmap=cmap, norm=LogNorm(vmin=1e-2, vmax=1e2))
    plt.show()
    
    # The numbers on the left side are given by the shape of the given image
    x_min_rank = x-rank if x-rank>0 else 0           #np.max([0, x-rank])
    x_max_rank = x+rank+1 if x+rank+1<img_shape[1] else img_shape[1] #np.min([1024, x+rank+1])
    y_min_rank = y-rank if y-rank>0 else 0           #np.max([0, y-rank])
    y_max_rank = y+rank+1 if y+rank+1<img_shape[0] else img_shape[0] #np.min([512, y+rank+1])
    
    peak_env = np.asarray(img[y_min_rank:y_max_rank, x_min_rank:x_max_rank], dtype=float)
    new_peak_env = np.copy(peak_env)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(new_peak_env, cmap=cmap, norm=LogNorm(vmin=1e-2, vmax=1e2))
    #peak_env_filtered = nd.median_filter(peak_env, size=2, mode='nearest')
    peak_env_filtered = nd.gaussian_filter(new_peak_env, sigma=0.5, mode='nearest')
    peak_env_filtered[peak_env_filtered<5e-2]=0
    peak_env_filtered[(peak_env_filtered>0) & (peak_env_filtered<1)]=1
    print(peak_env_filtered.mean(), np.all(new_peak_env==peak_env_filtered))
    ax2.imshow(peak_env_filtered, cmap=cmap, norm=LogNorm(vmin=1e-2, vmax=1e2))
    plt.show()
    
    peak_env_mask = new_peak_env<1
    new_peak_env += peak_env_filtered*peak_env_mask.astype(int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(new_peak_env, cmap=cmap, norm=LogNorm(vmin=1e-2, vmax=1e2))
    testing = ~peak_env_mask
    testing2 = testing.astype(int)
    print('Test: {}'.format(np.all(peak_env_filtered*testing2==peak_env)))
    
    peak_value = img[y][x]
    print('peak value: {}'.format(peak_value))
    
    peak_env_1d = new_peak_env.reshape(-1)             # 261 ns ± 9.17 ns
    
    # Look only at the current peak if there are other more intense peaks in the peak_env
    if peak_env_1d.max()!=peak_value:
        peak_env_1d = peak_env_1d[np.where(peak_env_1d<=peak_value)]
    
    # sorting it before calculating the histogram is faster
    sort_1d = np.sort(peak_env_1d)
    sort_1d_size = sort_1d.size-1
    q90 = sort_1d[int(np.round(sort_1d_size*0.9))]
    q10 = sort_1d[int(np.round(sort_1d_size*0.1))]
    median = sort_1d[int(np.round(sort_1d_size*0.5))]
    mean = peak_env_1d.mean()
    std = peak_env_1d.std()
    
    print('mean: {}, std: {}, median: {}, q10: {}, q90: {}'.format(mean, std, median, q10, q90))
    print('Ratio of mean/total_mean: {} and std/total_std: {}'.format(mean/total_mean, std/total_std))
    
    # int(number+1) is faster than int(np.ceil(number))      10.5 µs ± 585 ns compared to 103 ns ± 4.97 ns
    rel_size = new_peak_env.size/(2*rank + 1)**2
    peak_env_threshold = int( mean + 1)
    print('Threshold: {}'.format(peak_env_threshold))
    
    
    new_peak_env[new_peak_env<peak_env_threshold]=0
    
    labeled_peak_env, _ = nd.label(new_peak_env, structure=getStructure())#np.ones((3,3), dtype=int) 247 µs ± 2.86 µs
    x_in_peak_env = x if x<rank else rank       #np.min([x, rank])
    y_in_peak_env = y if y<rank else rank       #np.min([y, rank])
    peak_label = labeled_peak_env[y_in_peak_env, x_in_peak_env]
    if peak_label==0: 
        print('Not choosen:', peak)
    npix = np.sum((labeled_peak_env==peak_label) & (peak_env>=peak_env_threshold))
    integrated = nd.sum_labels(peak_env, labels=labeled_peak_env, index=peak_label)
    cont = labeled_peak_env==peak_label
    print('Npix: {}, I: {}, I/pixel: {}'.format(npix, integrated, integrated/npix))
    
    tmp = ax2.imshow(peak_env, cmap=cmap, norm=LogNorm(vmin=1e-2, vmax=1e2))
    divider1 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(tmp, cax=cax1)
    #ax.contour(labeled_peak_env, levels=1, cmap='jet')
    tmp2 = ax2.contour(cont, levels=1, colors=['b'], linewidths=[1])
    ax2.set_title('Npix: {}, I: {}, I/pixel: {}'.format(npix, integrated, integrated/npix))
    plt.show()
    
    original_npix_min = npix_min
    if rel_size < 1:
        npix_min = int( npix_min * (rel_size)**(1/6) )
        print('Size of the peak env: {}'.format(int(rel_size*(2*rank+1)**2)))
        print('New npix_min {} for peak {}'.format(npix_min, peak))