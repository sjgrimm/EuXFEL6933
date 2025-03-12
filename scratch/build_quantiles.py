
# -*- coding: utf-8 -*-

'''
Created on Friday 28.02.2025

@author: Jan Niklas Leutloff
'''

import argparse

import sys
sys.path.append('/UserData/kuschel/2025AuNP/analysis_tools')
from analysistools import data_helper as dh
from analysistools import azimuthalintegration as az

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import multiprocessing as mp

ffmpeg_path = '/arcv1/UserData/kuschel/2025AuNP/analysis_tools/ffmpeg/ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

#============================================================================================================================================

def drawWithoutReplacement(lst_in, numToDraw=1000):
    '''
    Parameters
    ----------
    lst_in : lst
        Input list

    numToDraw : int, optional
        Number of elements of lst_in that should be drawn randomly. Default is 1000.

    Returns
    -------
    Sorted list of random elements
    '''
    
    lst_process = np.copy(lst_in).tolist()
    
    if numToDraw >= len(lst_process): return lst_process
    
    lst_out = []
    for i in range(numToDraw):
        p = np.random.randint(0, len(lst_process))
        lst_out.append(lst_process.pop(p))
        
    return sorted(lst_out)

def selectRandomEvents(run, numToDraw=1000):

    '''
    Parameters
    ----------
    lst_in : lst
        Input list containing a list of tags.

    numToDraw : int, optional
        Number of tags that should be drawn randomly. Default is 1000.

    Returns
    -------
    Sorted list of random tags and the corresponding tags
    '''

    tags = dh.getTags(run)
    
    selectedEvents = drawWithoutReplacement(tags, numToDraw=numToDraw)
        
    data = dh.data_source(run)

    j=0
    list_img = []
    
    for img, tag in data:
        if tag==selectedEvents[j]:
            img[img<0] = 0
            list_img.append(np.copy(img))
            j+=1
        if j==len(selectedEvents): break
    
    return np.array(list_img), np.asarray(selectedEvents, dtype=int)

def sortImageStackPerPixel(list_img, axis=0):
    '''
    Parameters
    ----------
    lst_img : lst
        List of images in which each pixel is sorted in dependance of its brightness.

    axis: int, optional
        Axis along which the images should be sorted. Default is 0.

    Returns
    -------
    Pixel sorted images
    '''
    return np.sort(list_img, axis=axis)

def pickQuantiles(list_img):
    '''
    Parameters
    ----------
    lst_img : lst
        List of images in which each pixel is sorted in dependance of its brightness.

    Returns
    -------
    Quantile images and the corresponding quantiles
    '''
    
    length = len(list_img)
    
    start_quantile = int(np.round(length*0.99))
    steps = np.arange(start_quantile, length, 1)
    quantile = list(np.linspace(0.99, 1, length - start_quantile))
    
    quantile_0_5 = list_img[int(np.round(length*0.9))]
    quantile_0_9 = list_img[int(np.round(length*0.9))]
    quantile_0_95 = list_img[int(np.round(length*0.95))]
    
    pre_quantile = [0.5, 0.9, 0.95]
    ret_quantile = np.asarray(pre_quantile+quantile)
    
    ret = []
    ret.append(quantile_0_5)
    ret.append(quantile_0_9)
    ret.append(quantile_0_95)
    for step in steps:
        tmp = list_img[int(np.round(step))]
        ret.append(np.copy(tmp))
    
    return ret, ret_quantile

#========================================================================================================================================

def image_delivery(images, quantiles):
    for image, quantile in zip(images, quantiles):
        yield image, quantile

def main():
    parser = argparse.ArgumentParser(description='Ein Beispiel-Skript zum Parsen von Parametern.')

    parser.add_argument('--run', type=int, required=True, help='Die Run-Nummer')
    parser.add_argument('--energy', type=int, required=True, help='Die Energie')

    args = parser.parse_args()
    run = args.run
    energy = args.energy
    #run=int(sys.argv[1])
    
    str_run = str(run)
    
    random_images, selectedEvents = selectRandomEvents(run, numToDraw=1500)
    print('Done')
    
    sorted_images = sortImageStackPerPixel(random_images, axis=0)
    print('Done')
    
    quantile_images, quantiles = pickQuantiles(sorted_images)
    
    result = []

    
    #####################
    # COLORBAR SETTINGS #
    #####################
    v_min = 3e2
    v_max = 1e5
    #####################
    
    
    for q_img, q in zip(quantile_images, quantiles):
        result.append([q, q_img])
    result.append(selectedEvents)
        
    result = np.asarray(result, dtype=object)
        
    path = dh.result_path+'Quantiles/QuantileImages/'
    np.save(path+'run_'+str_run+'quantile_images', result)

    
    ############################
    # Maximal and median image #
    ############################
    
    fig, ax = plt.subplots(dpi=150)

    cmap = plt.get_cmap('inferno')
    cmap.set_under(color='black')  # Color for values below vmin (black)
    cmap.set_bad(color='black')

    max_img = ax.imshow(quantile_images[-1], norm=LogNorm(vmin=v_min, vmax=v_max), cmap=cmap)
    ax.set_title('Run {}: quantile 1.0'.format(run))
    ax.set_xlabel('x in pixel')
    ax.set_ylabel('y in pixel')
    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(max_img, cax=cax1)
    cb.set_label(label='Intensity in adu', labelpad=5)
    
    plt.tight_layout()
    plt.savefig(dh.result_path+'Quantiles/Maximal/run_{}_q1.0_{}eV.png'.format(run, energy))

    fig, ax = plt.subplots(dpi=150)

    cmap = plt.get_cmap('inferno')
    cmap.set_under(color='black')  # Color for values below vmin (black)
    cmap.set_bad(color='black')

    med_img = ax.imshow(quantile_images[0], norm=LogNorm(vmin=v_min, vmax=v_max), cmap=cmap)
    ax.set_title('Run {}: quantile 0.5'.format(run))
    ax.set_xlabel('x in pixel')
    ax.set_ylabel('y in pixel')
    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(med_img, cax=cax1)
    cb.set_label(label='Intensity in adu', labelpad=5)
    
    plt.tight_layout()
    plt.savefig(dh.result_path+'Quantiles/Median/run_{}_q0.5_{}eV.png'.format(run, energy))

    
    ##########################################
    # Radial profiles of the quantile images #
    ##########################################
    
    profiles = []
    for q_img in quantile_images:
        profiles.append(az.radialProfile(q_img, center=(1487, 532)))
    
    fig, ax = plt.subplots(dpi=150)
    cmap = plt.colormaps['jet']
    colors = cmap(np.linspace(0, 1, len(quantiles)))

    for p, profile in enumerate(profiles):
        ax.plot(profile, label=str(np.round(quantiles[p], 4)), color=colors[p])

    ax.set_title('Run {}: E = {} eV'.format(run, energy))
    ax.set_xlabel('x in pixel')
    ax.set_ylabel('Intensity in adu per pixel')
    ax.set_yscale('log')
    ax.set_ylim(1, 1e5)
    ax.legend(loc='lower left', ncol=6, fontsize=7)
    ax.grid()
    
    plt.tight_layout()
    plt.savefig(dh.result_path+'Quantiles/QuantileRadialProfile/run_{}_qrp.png'.format(run))


    ####################
    # Start with Video #
    ####################

    fig, ax = plt.subplots(dpi=300)
    deliverer = image_delivery(quantile_images, quantiles)
    shape = np.shape(quantile_images[0])
    img = np.ones(shape)                                      # In the resulting video the first two images wasn't shown
    quantile = -1                                             # So one has to initialize the video with something else -> don't call next
                                                              # but then also initialize quanitle with something else
    cmap = plt.get_cmap('inferno')
    cmap.set_under(color='black')  # Color for values below vmin (black)
    cmap.set_bad(color='black')

    global start_img
    start_img = ax.imshow(img, norm=LogNorm(vmin=v_min, vmax=v_max), cmap=cmap)
    ax.set_title('run: ' + str_run + ', quantile: ' + str(np.round(quantile, 4)))
    ax.set_xlabel('x in pixel')
    ax.set_ylabel('y in pixel')
    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(start_img, cax=cax1)
    cb.set_label(label='Intensity in adu', labelpad=5)
    
    def init_func():             # Initialize the update function to prevent missing images (also initialize quantile with -1)
        global start_img
        return (start_img.set_data(np.ones(shape)), ax.set_title('run: ' + str_run + ', quantile: ' + str(np.round(-1, 4))))

    def update(frame):
        global start_img
        img, quantile = next(deliverer)
        return (start_img.set_data(img), ax.set_title('run: ' + str_run + ', quantile: ' + str(np.round(quantile, 4))))

    path = dh.result_path+'Quantiles/QuantileVideos/'
    filename=path+'run_'+str_run+'_quantile_video.mp4'
    import os 
    if os.path.exists(filename):
        os.remove(filename)  
    
    ani = animation.FuncAnimation(fig=fig, init_func=init_func, func=update, frames=len(quantile_images), interval=500)
    ani.save(filename=filename, 
             writer=animation.FFMpegWriter(fps=1, codec='h264', extra_args=["-preset", "slower",
                                                                            "-crf", "21",
                                                                            "-profile:v", "baseline",
                                                                            "-level", "4.1"]))


    print('DONE!')
    return

if __name__ == '__main__':
    main()