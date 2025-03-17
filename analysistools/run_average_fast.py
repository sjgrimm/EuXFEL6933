import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools')
import data_helper as dh
import preselecting_data as pre_d
import xarray
import dask
import numpy as np
import matplotlib.pyplot as plt
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, progress
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

from damnit import Damnit
from extra_data import open_run
from extra_data.components import AGIPD1M

cmap = plt.get_cmap('inferno')
cmap.set_under(color='black')
cmap.set_bad(color='black')
norm = LogNorm(vmin=0.0000128205,vmax=0.103728)
proposal =dh.proposal

avg_path = '/gpfs/exfel/u/usr/SPB/202501/p006933/Results/RunAverages/flag_nshot{}_r{}_average.npy'
max_path = '/gpfs/exfel/u/usr/SPB/202501/p006933/Results/RunMax/flag_nshot{}_r{}_max.npy'

def compute_average(img, flag_dict):
    return img.sel(flag_dict).mean('train_pulse')

def compute_max(img, flag_dict):
    return img.sel(flag_dict).max('train_pulse')

class Analysis_flag:
    def __init__(self, run, nshot):
        self.run_number = run
        self.run = open_run(proposal=proposal, run=self.run_number, parallelize=True)
        self.agipd = AGIPD1M(self.run)
        self.nshot = nshot
        self.df_flags = pre_d.Run(self.run_number).reduced_data
        self.train_list = self.df_flags['trainId'].to_numpy()
        self._img_dask = None
        self._filter_dict = None

    @property
    def filter_dict(self):
        if self._filter_dict is None:
            result=[]
            used_shots=0
            for t_id in self.train_list:
                if used_shots>=self.nshot: break
                train_df=self.df_flags.loc[self.df_flags['trainId'] == t_id]
                if len(train_df) >= self.nshot-used_shots: selected_entries = train_df.head(self.nshot-used_shots)
                else: selected_entries = train_df
                for _, row in selected_entries.iterrows():
                    result.append((row['trainId'], row['pulseId']))
                used_shots+=len(train_df)
            self._filter_dict={'train_pulse':result}
        
        return self._filter_dict
        
    @property
    def img_dask(self):
        if self._img_dask is None:
            self._img_dask = self.agipd.get_dask_array('image.data')
        return self._img_dask

def main():
    start_run=int(sys.argv[1])
    end_run=int(sys.argv[2])
    nshot=int(sys.argv[3])
    log_directory = f"/gpfs/exfel/exp/SPB/202501/p006933/usr/Shared/muthreich/Logs/"
    os.makedirs(log_directory, exist_ok=True)
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
    
        analysis = [Analysis_flag(i, nshot=nshot) for i in range(start_run,end_run+1)]
        results = [(compute_average(ana.img_dask, ana.filter_dict), compute_max(ana.img_dask, ana.filter_dict)) for ana in analysis]
        results = dask.compute(*results)

    for result, ana in zip(results, analysis):
        agipd_geom = dh.getGeometry(ana.run_number)
        assem_avg, _ = agipd_geom.position_modules(result[0])
        assem_max, _ = agipd_geom.position_modules(result[1])

        
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        im=ax.imshow(assem_avg, norm=norm, cmap=cmap)
        ax.set_title(f'Average Run {ana.run_number} Image')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  
        cbar = fig.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(f'/gpfs/exfel/exp/SPB/202501/p006933/usr/Results/RunAverages/flag_nshot{ana.nshot}_r{ana.run_number}_avergae.png', dpi=300,format='png')
        plt.close()
        
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        im=ax.imshow(assem_max, norm=LogNorm(vmin=1e-2, vmax=np.max(assem_max)), cmap=cmap)
        ax.set_title(f'Max Run {ana.run_number} Image')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  
        cbar = fig.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(f'/gpfs/exfel/exp/SPB/202501/p006933/usr/Results/RunMax/flag_nshot{ana.nshot}_r{ana.run_number}_max.png', dpi=300,format='png')
        plt.close()

        
        np.save(avg_path.format(ana.nshot, ana.run_number), result[0])
        np.save(max_path.format(ana.nshot, ana.run_number), result[1])

if __name__ == '__main__':
    t0=time.time()
    main()
    t1=time.time()
    print(f'finished in {t1-t0} seconds', flush=True)