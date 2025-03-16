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
import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools')
import data_helper as dh
import preselecting_data as pre_d
import argparse
import pandas as pd

from damnit import Damnit
from extra_data import open_run
from extra_data.components import AGIPD1M

from tqdm import tqdm

path = dh.expPath+'Results/FocusScans/DataDask/'

proposal = 6933

def mask_full_fluor(bad=False):
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

class Analysis:
    def __init__(self, run, att, energy, z_pos_list, train_list, nshot, df_flags, flag):
        self.nrun = run
        self.run = open_run(proposal=proposal, run=self.nrun, parallelize=True)
        self.agipd = AGIPD1M(self.run)
        self.att=att
        self.energy=energy
        self.z_pos_list = z_pos_list
        self.train_list = train_list
        self.nshot = nshot
        self.df_flags = df_flags
        self.flag=flag
        self._average = None

    def create_filterdict(self):
        result=[]
        for train_list in self.train_list:
            used_shots=0
            for t_id in train_list:
                if used_shots>=self.nshot: break
                train_df=self.df_flags.loc[self.df_flags['trainId'] == t_id]
                filtered_df =  train_df.loc[self.df_flags['flags'] == 1]
                if len(filtered_df) >= self.nshot-used_shots: selected_entries = filtered_df.head(self.nshot-used_shots)
                else: selected_entries = filtered_df
                for _, row in selected_entries.iterrows():
                    result.append((row['trainId'], row['pulseId']))
                used_shots+=len(filtered_df)
        return {'train_pulse':result}
        
    @property
    def average(self):
        if self._average is None:
            img = self.agipd.get_dask_array('image.data')
            flag_dict=self.create_filterdict()
            self._average = img.sel(flag_dict).mean('train_pulse')
        

        return self._average

def main(run=None, flag_num=1, nshot=200):
    mask=mask_full_fluor()
    analysis=[]
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
    
    df=pre_d.Run(run).reduced_data

    for att in df['total_transmission'].unique():
        df_att = df[df['total_transmission']==att]
        
        for energy in df_att['photon_energy'].unique():
            df_e = df_att[df_att['photon_energy']==energy]

            pos_list = sorted(df_e['inj_pos_z'].unique())
            new_pos_list = []
            train_list = []
            
            for pos in pos_list:
                train_array = df_e[df_e['inj_pos_z']==pos]['trainId'].to_numpy()

                if len(train_array)<20: continue

                print('Position and their number of trains:', pos, len(train_array))
                new_pos_list.append(pos)
                train_list.append(train_array)

            analysis.append(Analysis(run, att, energy, new_pos_list, train_list, nshot, df, flag)) 
 
    log_directory = f"/gpfs/exfel/exp/SPB/202501/p006933/usr/Shared/muthreich/Logs/"
    os.makedirs(log_directory, exist_ok=True)

    # Define the SLURMCluster with the log directory
    cluster_kwargs = {
    'queue': 'upex',
    'local_directory': '/scratch',
    'processes': 16,
    'cores': 16,
    'memory': '512GB',
    'log_directory': log_directory,  # Set the log directory
    }

    with SLURMCluster(**cluster_kwargs) as cluster, Client(cluster) as client:
        cluster.scale(32)
        results = [(ana.average) for ana in analysis]
        results = dask.compute(*results)


    #Hier rufe ich die schleife von oben zum speichern nochmal auf. das geht bestimmt besser so ist es aber gerade am einfachsten und schnellsten für mich
    i=0
    for att in df['total_transmission'].unique():
        df_att = df[df['total_transmission']==att]
        
        for energy in df_att['photon_energy'].unique():
            df_e = df_att[df_att['photon_energy']==energy]

            pos_list = sorted(df_e['inj_pos_z'].unique())
            new_pos_list = []
            train_list = []
            
            for pos in pos_list:
                train_array = df_e[df_e['inj_pos_z']==pos]['trainId'].to_numpy()

                if len(train_array)<20: continue

                print('Position and their number of trains:', pos, len(train_array))
                new_pos_list.append(pos)
                train_list.append(train_array)

            #currently not working for multiple z-positions
            ret_dict = {}
            ret_dict['transmission'] = att
            ret_dict['photon_energy'] = energy
            ret_dict['inj_pos_z'] = new_pos_list
            ret_dict['run']=run
            ret_dict['f_yield'] = float(np.mean(results[i]*mask))
            i+=1
            df_ret = pd.DataFrame(ret_dict)

            
            path = dh.expPath+'Results/FocusScans/Data/'
            df_ret.to_hdf(path+'fyield_r{}_att{}_{}eV_ROIs_n{}.h5'.format(dh.run_format(run), att, energy, nshot), key='f_yield')


    
    return

if __name__ == '__main__':
    main()
    print('finished', flush=True)
