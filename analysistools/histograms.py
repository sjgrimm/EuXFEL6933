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

def mask_box():
    mask = np.zeros((16, 512, 128))
    mask[7][0:256, 0:128]=1
    mask[11][128+64:512, 0:128]=1
    return mask

def calcLitPixel(run, trains, mask):
    
    lit_pix=[]
    lit_pix_mask=[]
    #lit_pix_norm=[]
    #lit_pix_mask_norm=[]
    data = dh.pulse_source(run=run, train_list=trains, flag=True)

    for t_id, p_id, img in data:
        lit_pix.append(np.sum(img>dh.f_threshold))
        lit_pix_mask.append(np.sum(img>dh.f_threshold, where=mask==1))
        
    return [lit_pix, lit_pix_mask]

def histogram(run, att, energy, train_array, save=False):

    mask = mask_box()
    parameter = []
    for train in train_array:
        parameter.append([run, [train], mask])

    nworker = np.min([32, len(train_array)])
    with Pool(nworker, maxtasksperchild=1) as pool:
        results = pool.starmap(calcLitPixel, parameter)

    ret_pix = []
    ret_pix_mask = []
    for r in results:
        ret_pix.extend(r[0])
        ret_pix_mask.extend(r[1])

    print(len(ret_pix), len(ret_pix_mask))

    ret_dict = {}
    ret_dict['lit_pix'] = ret_pix
    ret_dict['lit_pix_mask'] = ret_pix_mask
    df = pd.DataFrame(ret_dict)
    if save:
        path = dh.expPath+'Results/Histograms/Data/'
        df.to_hdf(path+'hist_r{0:}_{1:.2%}_{2:}eV_wao_mask.h5'.format(dh.run_format(run), att, energy), key='hist')

    limit=6500
    plt.figure(dpi=150)
    plt.hist(ret_pix, bins=100, range=(0,limit))#, label='{} eV'.format(energy))
    plt.grid()
    #plt.legend()
    plt.xlabel('Number of lit pixels')#$\mathrm{\mu}$m')
    plt.ylabel('Number of bunches')
    plt.title('Run {0:} at {1:} eV and {2:.2%} transmission without mask'.format(run, energy, att))
    path = dh.expPath+'Results/Histograms/'
    if save: 
        plt.savefig(path+'hist_r{0:}_{1:.2%}_{2:}eV_all.png'.format(dh.run_format(run), att, energy))
    plt.show()

    plt.figure(dpi=150)
    plt.hist(ret_pix_mask, bins=100, range=(0,limit))#, label='{} eV'.format(energy))
    plt.grid()
    #plt.legend()
    plt.xlabel('Number of lit pixels')#$\mathrm{\mu}$m')
    plt.ylabel('Number of bunches')
    plt.title('Run {0:} at {1:} eV and {2:.2%} transmission with mask'.format(run, energy, att))
    path = dh.expPath+'Results/Histograms/'
    if save: 
        plt.savefig(path+'hist_r{0:}_{1:.2%}_{2:}eV_mask.png'.format(dh.run_format(run), att, energy))
    plt.show()
    return

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
            train_array = df_e['trainId'].to_numpy()

            histogram(run, att, energy, train_array, save=True)
            
    
    return

if __name__ == '__main__':
    main()