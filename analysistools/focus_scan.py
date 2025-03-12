'''
Created on Friday 28.02.2025

@author: Jan Niklas Leutloff

|---------------------------------------|
| Look first if the mask still works!!! |
|---------------------------------------|
'''

import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/scratch')
import data_helper as dh
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/generatorpipeline')
from generatorpipeline.generatorpipeline import accumulators as acc

import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import argparse

def mask_img_ring(radius_in=0, radius_out=0.05):
    '''
    Returns a ring mask.
    '''
    geom = dh.getGeometry()
    pixpos = geom.get_pixel_positions()
    px, py, pz = np.moveaxis(pixpos, -1, 0)  # Separate x, y, z coordinates
    #px.shape  # (modules, slow scan, fast scan)
    radius = np.sqrt(px**2 + py**2)

    mask = (radius_in < radius) & (radius < radius_out)
    
    return mask

def mask_corner(run, corner, border=150):
    '''
    Returns a ring mask.
    '''
    
    data = dh.data_source(run)
    img, tag = next(data)
    y_max, x_max = img.shape
    mask = np.zeros((y_max, x_max), dtype=int)

    if corner==1:
        mask[0:border+1,0:border+1] = 1
    elif corner==2:
        mask[0:border+1,x_max-border-1:x_max+1] = 1
    elif corner==3:
        mask[y_max-border-1:y_max+1,0:border+1] = 1
    elif corner==4:
        mask[y_max-border-1:y_max+1,x_max-border-1:x_max+1] = 1
    
    return mask

def mask_box(run, x_min, x_max, y_min, y_max):

    data = dh.data_source(run)
    img, tag = next(data)
    y_s, x_s = img.shape
    mask = np.zeros((y_s, x_s), dtype=int)

    mask[y_min:y_max, x_min:x_max]=1

    return mask

def mask_stripe(run, y_lower=0, y_upper=150):
    '''
    Returns a stripe mask (stripe along x).
    '''

    data = dh.data_source(run)
    img, tag = next(data)
    y_max, x_max = img.shape
    mask = np.zeros((y_max, x_max), dtype=int)

    mask[y_lower:y_upper+1, 0:x_max] = 1
    
    return mask

def calcFluorescence(run, shotN=500):
    '''
    Parameters
    ----------
    run : int
        The run number
        
    border : int, optional
        The size of the mask
    
    shotN : int, optional
        Number of images which are should contribute to the fluorescence calculation

    Returns
    -------
    Fluorescence per pulse energy of the run: float
    '''
    masks = []
    mask = mask_img_ring()
    masks.append(mask)
    #mask0 = mask_stripe(run, y_lower=50, y_upper=200)
    #mask1 = mask_corner(run, corner=1)
    #mask2 = mask_corner(run, corner=2)
    #mask3 = mask_corner(run, corner=3)
    #mask4 = mask_corner(run, corner=4)
    #mask5 = mask_stripe(run)
    #mask6 = mask_stripe(run, y_lower=0, y_upper=50)
    #mask7 = mask_box(run, 320, 720, 500, 600)
    #mask8 = mask_box(run, 320, 720, 800, 900)

    data = dh.pulse_source(run)
    #pulseEnergies = dh.getPulseEnergy(tags, highTag)
    
    j=0
    fluorescence = []
    for _ in range(len(masks)):
        fluorescence.append(acc.Mean())

    for t_id, p_id, img in data:
        if j>shotN: break

        #pulseEnergy = pulseEnergies[i]
        #if pulseEnergy==float('nan') or pulseEnergy==0: continue
        
        if np.sum(img>dh.f_threshold)<50: continue
        img[img<0]=0 # warum?
        
        j+=1

        for i, mask in enumerate(masks):
            mean = img[mask==1].mean()
            fluorescence[i].accumulate(mean)

    print('Done with run {}'.format(run), flush=True)
    if len(fluorescence[0])==0: return 0
    if len(len(fluorescence)==1): return fluorescence[0].value
    return [accum.value for accum in fluorescence]

def findFocus(run_list, start_run, num_run, energy, save=False):
    '''
    Parameters
    ----------
    run_list : list
        List with runs for which the focus scan should be applied
        
    Returns
    -------
    np.ndarray that contains the fluorescence corresponding to the runs
    '''
    
    with Pool(len(run_list), maxtasksperchild=1) as pool:
        results = pool.map(calcFluorescence, run_list)
    
    mean_fluorescence = np.asarray(results, dtype=object)

    if save:
        path = dh.expPath+'Results/FocusScans/Data/'
        np.save(path+'fyield_{}eV_r{}_r{}_ROIs'.format(energy, start_run, start_run+num_run-1), mean_fluorescence)

    print('DONE!', flush=True)
    
    return mean_fluorescence

def Gauss(x, a, center, sigma, c):
    return a * np.exp(-((x-center)**2)/(2*sigma**2) ) + c

def fitFocus_scipy(target_thickness, scanned_fyield, start_run, num_run, filename_ext, save=False, fit=False, energy=None):
    '''
    Parameters
    ----------
    target_thickness : nm
    ys : array-like
        List of y positions

    scanned_fyield : array-like
        Fluorescence at the y positions
        
    Returns
    -------
    np.ndarray that contains the fluorescence corresponding to the runs
    '''

    if energy is None:
        energy=dh.getPhotonEnergy(start_run)
    #attn = dh.getAttenuatorSiliconThickness(tags[0], highTag)

    zs = []
    for i in range(start_run, start_run+num_run):
        prof_y_position=
        zs.append(prof_y_position[0])
    
    data = np.stack([ys, scanned_fyield])
    data = data[:,data[0].argsort()]

    if fit:
        p,c=curve_fit(Gauss, data[0], data[1], 
                      bounds=([0.001,
                               np.max([ys[np.argmax(scanned_fyield)]-5e2, np.min(ys)]),   2,  0], 
                              [np.max(scanned_fyield)*2, 
                               np.min([ys[np.argmax(scanned_fyield)]+5e2, np.max(ys)]), 500, 50]))
        x_fit = np.arange(np.min(data[0]), np.max(data[0]))
        focus_fit = Gauss(x_fit, *p)

    plt.figure(dpi=150)
    if fit: plt.plot(x_fit, focus_fit, color='tab:orange', label='Fit')
    plt.scatter(data[0], data[1], label='Data')
    plt.grid()
    plt.legend()
    plt.xlabel(r'y position in $\mathrm{\mu}$m')
    plt.ylabel('Fluorescence yield in photons per pixel')
    plt.title('Au {} nm: y scan at {} eV and attn {} mm'.format(target_thickness, int(energy), attn))
    path = dh.expPath+'Results/FocusScans/'
    if save: plt.savefig(path+'focusScan_Au{}nm_{}eV_{}mm_r{}_r{}_ROI{}.png'.format(target_thickness, int(energy), attn, start_run, start_run+num_run-1, str(filename_ext)))
    plt.show()
    
    return

#========================================================================================================================

def main():

    parser = argparse.ArgumentParser(description='Ein Beispiel-Skript zum Parsen von Parametern.')

    parser.add_argument('--run', type=int, required=True, help='Run number')
    parser.add_argument('--energy', type=int, required=True, help='Energy')
    parser.add_argument('--thickness', type=int, required=True, help='Target thickness')

    args = parser.parse_args()
    run = args.run
    energy = args.energy
    target_thickness = args.thickness
    
    fluorescence_list = findFocus(run_list, start_run, num_run, energy, save=True)

    if np.ndim(fluorescence_list)==1:
        fitFocus_scipy(target_thickness, fluorescence_list, start_run, num_run, 0, save=True, energy=energy)
    else:
        for i in range(np.ndim(fluorescence_list)):
            fitFocus_scipy(target_thickness, fluorescence_list[:,i], start_run, num_run, i, save=True, energy=energy)
    
    return

if __name__ == '__main__':
    main()