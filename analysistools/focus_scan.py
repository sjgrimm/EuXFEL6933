'''
Created on Friday 28.02.2025

@author: Jan Niklas Leutloff

|---------------------------------------|
| Look first if the mask still works!!! |
|---------------------------------------|
'''

import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools')
import data_helper as dh
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/generatorpipeline')
import generatorpipeline.accumulators as acc

import pandas as pd
import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import argparse

def mask_img_ring(run, radius_in=0, radius_out=0.05):
    '''
    Returns a ring mask.
    '''
    geom = dh.getGeometry(run)
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

def mask_box():
    '''
    12 | 0
    13 | 1
    14 | 2
    15 | 3
    ---+---
     8 | 4
     9 | 5
    10 | 6
    11 | 7
    '''
    
    mask = np.zeros((16, 512, 128))
    mask[1][0:128, 0:128]=1
    mask[10][0:128, 0:128]=1

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

def calcFluorescence(run, trains, shotN=250):
    '''
    Parameters
    ----------
    run : int
        The run number
        
    trains : list of integers
        Contains the train_ids that should be passed to the pulse_source
        
    shotN : int, optional
        Number of images which are should contribute to the fluorescence calculation

    Returns
    -------
    Fluorescence per pulse energy of the run: float
    '''
    masks = []
    mask = mask_box()
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

    data = dh.pulse_source(run=run, train_list=trains, flag=False)
    #pulseEnergies = dh.getPulseEnergy(tags, highTag)
    
    j=0
    fluorescence = []
    for _ in range(len(masks)):
        fluorescence.append(acc.Mean())

    for t_id, p_id, img in data:
        if j>shotN: break

        #pulseEnergy = pulseEnergies[i]
        #if pulseEnergy==float('nan') or pulseEnergy==0: continue
        
        if np.sum(img>dh.f_threshold)<150: continue
        img[img<0]=0 # warum?
        
        j+=1

        for i, mask in enumerate(masks):
            mean = img[mask==1].mean()
            fluorescence[i].accumulate(mean)

    print('Done with run {}'.format(run), flush=True)
    #if len(len(fluorescence)==1): return fluorescence[0].value
    return [accum.value for accum in fluorescence]

def z_scan_fluorescence(run, att, energy, pos_list, train_list, save=False):
    '''
    Parameters
    ----------
    run_list : list
        List with runs for which the focus scan should be applied
        
    Returns
    -------
    np.ndarray that contains the fluorescence corresponding to the runs
    '''
    parameter = []
    for trains in train_list:
        parameter.append([run, trains])
    
    with Pool(len(pos_list), maxtasksperchild=1) as pool:
        results = pool.starmap(calcFluorescence, parameter)
    
    mean_fluorescence = np.asarray(results, dtype=float)
    print(np.shape(mean_fluorescence))

    ret_dict = {}
    ret_dict['transmission'] = att
    ret_dict['photon_energy'] = energy
    ret_dict['injector_pos'] = pos_list
    for i in range(np.shape(mean_fluorescence)[1]):
        ret_dict['f_yield_ROI{}'.format(i)] = mean_fluorescence[:,i]
    
    df = pd.DataFrame(ret_dict)

    if save:
        path = dh.expPath+'Results/FocusScans/Data/'
        df.to_hdf(path+'fyield_r{}_att{}_{}eV_ROIs.h5'.format(dh.run_format(run), att, energy), key='f_yield')

    print('DONE with z_scan for {} eV!'.format(energy), flush=True)
    
    return df

def Gauss(x, a, center, sigma, c):
    return a * np.exp(-((x-center)**2)/(2*sigma**2) ) + c

def find_focus_scipy(run, att, energy, df_fluorescence, save=False, fit=False):
    '''
    Parameters
    ----------
    target_thickness : int
        Nanoparticle size in nm
        
    z_s : array-like
        List of z positions
        
    scanned_fyield : array-like
        Fluorescence at the z positions
        
    Returns
    -------
    np.ndarray that contains the fluorescence corresponding to the runs
    '''
    z_s = df_fluorescence['injector_pos'].to_numpy()
    for i in range(df_fluorescence.shape[1]-3):
        scanned_fyield = df_fluorescence['f_yield_ROI{}'.format(i)].to_numpy()
    
        data = np.stack([z_s, scanned_fyield])
        data = data[:,data[0].argsort()]
    
        if fit:
            p,c=curve_fit(Gauss, data[0], data[1], 
                          bounds=([0.001,
                                   np.max([z_s[np.argmax(scanned_fyield)]-5e2, np.min(z_s)]),   2,  0], 
                                  [np.max(scanned_fyield)*2, 
                                   np.min([z_s[np.argmax(scanned_fyield)]+5e2, np.max(z_s)]), 500, 50]))
            x_fit = np.arange(np.min(data[0]), np.max(data[0]))
            focus_fit = Gauss(x_fit, *p)
    
        plt.figure(dpi=150)
        if fit: plt.plot(x_fit, focus_fit, color='tab:orange', label='Fit')
        plt.scatter(data[0], data[1], label='Data')
        plt.grid()
        plt.legend()
        plt.xlabel(r'z position in mm')#$\mathrm{\mu}$m')
        plt.ylabel('Fluorescence yield in photons per pixel')
        plt.title('Run {0:}: z scan at {1:} eV and {2:.2%} transmission'.format(run, energy, att))
        path = dh.expPath+'Results/FocusScans/'
        if save: 
            plt.savefig(path+'focusScan_r{0:}_{1:.2%}_{2:}eV_ROI{3:}.png'.format(dh.run_format(run), att, energy, i))
        plt.show()
    
    return

#========================================================================================================================

def main(run=None):

    if run is not None:
        run = run
    else:
        parser = argparse.ArgumentParser(description='Ein Beispiel-Skript zum Parsen von Parametern.')
    
        parser.add_argument('--run', type=int, required=True, help='Run number')
        
        args = parser.parse_args()
        run = args.run
        
    df_e = dh.getPhotonEnergy_trainwise(run)
    df_p = dh.getInjectorPos_trainwise(run)
    df_att = dh.getTransmission_trainwise(run)
    df = pd.merge(df_e, df_p, on='trainId', how='inner')
    df = pd.merge(df, df_att, on='trainId', how='inner')

    for att in df['total_transmission'].unique():
        df_att = df[df['total_transmission']==att]
        
        for energy in df_att['photon_energy'].unique():
            df_e = df_att[df_att['photon_energy']==energy]

            pos_list = df_e['injector_pos'].unique()
            train_list = []
            
            for pos in pos_list:
                train_array = df_e[df_e['injector_pos']==pos]['trainId'].to_numpy()

                size = train_array.size
                middle = size//2
                lower = middle-1 if middle-1>0 else 0
                upper = middle+2 if middle+2<size else size
                tmp = train_array[lower:upper]
                train_list.append(tmp)
                print(train_array[middle], tmp)

            df_fluorescence = z_scan_fluorescence(run, att, energy, pos_list, train_list, save=True)
            find_focus_scipy(run, att, energy, df_fluorescence, save=True)
    
    return

if __name__ == '__main__':
    main()
