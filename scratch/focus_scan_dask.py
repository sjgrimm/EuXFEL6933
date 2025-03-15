'''
Created on Friday 15.03.2025

@author: Nils Muthreich

|------------------------------------------------------------------------------------------------------------------------------|
| Try to implement the focus scan with same output format as Jans focus_scan.py with dask arrays for better performance(maybe) |
|------------------------------------------------------------------------------------------------------------------------------|
'''
import xarray
import dask
import numpy as np
import matplotlib.pyplot as plt
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, progress

from damnit import Damnit
from extra_data import open_run
from extra_data.components import AGIPD1M

from tqdm import tqdm

proposal = 6933

def mask_full_flour(bad=False):
    '''
    plot_data:  12 | 0   imshow:  7 | 11
                13 | 1            6 | 10
                14 | 2            5 |  9
                15 | 3            4 |  8
                ---+---          ---+---
                 8 | 4            3 | 15
                 9 | 5            2 | 14
                10 | 6            1 | 13
                11 | 7            0 | 12
    '''
    import h5py
    with h5py.File(dh.expPath+'Shared/geom/mask_hvoff_20250311.h5', 'r') as f:
        bad_pixel = f['entry_1/data_1/mask'][:].astype(bool)

    if not bad:
        mask = np.zeros((16, 512, 128))
        mask[7][64:192, 0:128]=1
        mask[7][256:512, 0:128]=1
        mask[1][0:128, 0:128]=1
        mask[10][0:128, 0:128]=1
        mask[11][256:512, 0:128]=1
        mask[12][64:192, 0:128]=1
        mask = mask.astype(bool)
        mask = ~bad_pixel*mask
    else:
        mask = ~bad_pixel

    return mask

def main(run=None, flag_num=1, nshot=200):

    if run is not None:
        run = run
    else:
        parser = argparse.ArgumentParser(description='Ein Beispiel-Skript zum Parsen von Parametern.')
    
        parser.add_argument('--run', type=int, required=True, help='Run number')
        parser.add_argument('--flag', type=int, required=True, help='Wheather to flag (1) or not (0)')
        parser.add_argument('--nshot', type=int, required=True, help='Wheather to flag (1) or not (0)')
        
        args = parser.parse_args()
        run = args.run
        flag_num = args.flag
        nshot = args.nshot

    flag = True if flag_num==1 else False
        
    df_e = dh.getPhotonEnergy_trainwise(run)
    df_p = dh.getInjectorPos_trainwise(run)
    df_att = dh.getTransmission_trainwise(run)
    df_f = dh.getFlags(run)
    df = pd.merge(df_e, df_p, on='trainId', how='inner')
    df = pd.merge(df, df_att, on='trainId', how='inner')

    for att in df['total_transmission'].unique():
        df_att = df[df['total_transmission']==att]
        
        for energy in df_att['photon_energy'].unique():
            df_e = df_att[df_att['photon_energy']==energy]

            pos_list = sorted(df_e['injector_pos'].unique())
            new_pos_list = []
            train_list = []
            
            for pos in pos_list:
                train_array = df_e[df_e['injector_pos']==pos]['trainId'].to_numpy()

                if len(train_array)<20: continue

                print('Position and their number of trains:', pos, len(train_array))
                new_pos_list.append(pos)
                train_list.append(train_array)

            df_fluorescence = z_scan_fluorescence(run, att, energy, new_pos_list, train_list, nshot, df_f, flag, save=True)
            find_focus_scipy(run, att, energy, df_fluorescence, nshot, save=True)
    
    return

if __name__ == '__main__':
    main()
