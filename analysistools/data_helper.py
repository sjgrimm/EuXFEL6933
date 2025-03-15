'''
Created on Thursday 27.02.2025

@author: Jan Niklas Leutloff

import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools')
import data_helper as dh
'''

import os
import sys

import numpy as np
import pandas as pd
from multiprocessing import Pool

from tqdm import tqdm
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import curve_fit
import scipy.constants as cnst
from scipy.signal import find_peaks
import scipy.ndimage as nd
from skimage.feature import peak_local_max
from skimage.measure import profile_line

import extra_data as ex
from extra_geom import AGIPD_1MGeometry
from extra_geom.motors import AGIPD_1MMotors
from extra.components import AGIPD1MQuadrantMotors

#---------------------------------------------------------------------------------------------------------------
##############
# Experiment #
##############

expPath = '/gpfs/exfel/exp/SPB/202501/p006933/usr/'#/gpfs/exfel/u/usr/SPB/202501/p006933/Software
proposal = 6933

#---------------------------------------------------------------------------------------------------------------
#########################
# Sources and constants #
#########################

f_threshold = 0
e_offset = 120

det = {
    'hirex': 'SA1_XTD9_HIREX/CORR/GOTTHARD_RECEIVER:daqOutput',  # Spectrometer
    'xgm2': 'SA1_XTD2_XGM/XGM/DOOCS:output',                     # Gas detector in SASE1
    'att_xgm2': 'SA1_XTD2_ATT/MDL/MAIN',                         # Attenuation by xgm9
    'xgm9': 'SPB_XTD9_XGM/XGM/DOOCS:output',                     # Gas detector next to SPB
    'att_xgm9': 'SPB_XTD9_ATT/MDL/MAIN',                         # Attenuation by xgm9
    'agipd': 'SPB_DET_AGIPD1M-1/CORR/*',                         # Main 2d detector
    'agipd_z': 'SPB_IRU_AGIPD1M/MOTOR/Z_STEPPER',                # z position of AGIPD
    'inj_x': 'SPB_IRU_INJMOV/MOTOR/X',                           # positions of the injector
    'inj_y': 'SPB_IRU_INJMOV/MOTOR/Y',                           #
    'inj_z': 'SPB_IRU_INJMOV/MOTOR/Z',                           #
    'undulator_e': 'SPB_XTD2_UND/DOOCS/ENERGY',                  # Energy set by the undulator
    #'inj_cam_down': 'SPB_EXP_ZYLA/CAM/1:daqOutput',              # Injector nozzle camera
    #'inj_cam_up': 'SPB_IRU_AEROSOL/CAM/CAM_1:daqOutput',         # Injector nozzle camera
    'hitfinder': 'SPB_DET_AGIPD1M-1/REDU/SPI_HITFINDER:output',  # Flags the hits
    'frames': 'SPB_IRU_AGIPD1M1/REDU/LITFRM:output',
    #'test1': 'SPB_EHD_IBS/CAM/1:daqOutput',             # doesn't work
    #'test3': 'SPB_IRU_AGIPD1M1/REDU/LITFRM:output',     # weird
    #'test4': 'ACC_SYS_DOOCS/CTRL/BEAMCONDITIONS'        # kParameter
    #'test5': 'SPB_IRU_AGIPD1M1/MDL/DATA_SELECTOR',      # not usefull
    #'test6': 'SPB_IRU_AGIPD1M/PSC/HV',                  # not usefull
}

det_fast = {
    'hirex': 'SA1_XTD9_HIREX/CORR/GOTTHARD_RECEIVER:daqOutput',  # Spectrometer
    'xgm2': 'SA1_XTD2_XGM/XGM/DOOCS:output',                     # Gas detector in SASE1
    'att_xgm2': 'SA1_XTD2_ATT/MDL/MAIN',                         # Attenuation by xgm9
    'xgm9': 'SPB_XTD9_XGM/XGM/DOOCS:output',                     # Gas detector next to SPB
    'att_xgm9': 'SPB_XTD9_ATT/MDL/MAIN',                         # Attenuation by xgm9
}

det_image = {
    'agipd': 'SPB_DET_AGIPD1M-1/CORR/*',                         # Main 2d detector
    'hitfinder': 'SPB_DET_AGIPD1M-1/REDU/SPI_HITFINDER:output',  # Flags the hits
}

#---------------------------------------------------------------------------------------------------------------
##################################################
# Data source with all sources of the experiment #
##################################################

def data_source(run, verbose=False):
    '''
    Returns
    -------
    The a DataCollection with the trains that are included in all det sources.
    '''
    ds = ex.open_run(proposal=proposal, run=run)
    original_number_of_trains = len(ds.train_ids)
    elements = []
    for key in det.keys():
        try:
            tmp = ds.select([det[key]], require_all=True)
            if verbose:
                print('Source {0:} has {1:.2%} of the data'.format(key, len(tmp.train_ids)/original_number_of_trains), flush=True)
                if len(tmp.train_ids)==0: print(f'No data for {key}', flush=True)
            elements.append(det[key])
        except: 
            print(f"{key} not detected", flush=True)
    sel = ds.select(elements, require_all=True)
    number_of_trains = len(sel.train_ids)
    print('{0:.2%} of the data of run {1:} is used!'.format(number_of_trains/original_number_of_trains, run), flush=True)
    if number_of_trains==0: print('One or more sources did not collect data!', flush=True)
    return sel
    
#---------------------------------------------------------------------------------------------------------------
##################
# Image delivery #
##################

def train_source(run, verbose=False):
    '''
    Parameter
    ---------
    run : int
        The run number.
    fast : bool, optional
        If True the train data doesn't contain agipd data.
        Default is False.
    verbose : bool, optional
        Wheather to print debug information (True) or not (default: False).
        
    Yields
    ------
    The trainId as int and the corresponding data of the train as dict
    '''
    ds = data_source(run)
    if verbose:    
        ds.info()
        ds.all_sources

    elements = []
    for key in det_image.keys():
        try:
            ds.select(det_image[key])
            elements.append(det_image[key])
        except: 
            print(f"{key} not detected")
    sel = ds.select(elements)
    
    for t_id, t_data in sel.trains(require_all=True):
        yield t_id, t_data

def fast_data_source(run, verbose=False):
    '''
    Not implemented yet!
    '''
    return 


def pulse_source(run, train_list=None, flag=True, verbose=False):
    '''
    Parameter
    ---------
    run : int
        The run number
    train_list : list, optional
        If a list of train_ids is given then only the pulse data of these trains are yielded.
        Default is None.
    flag : bool, optional
        Wheather to yield only the hits (default: True) of not (False)
    verbose : bool, optional
        Wheather to print debug information (True) or not (default: False)
        
    Yields
    ------
    The trainId and the pulseId as int, respectively and the corresponding stacked image
    '''
    if train_list is None:
        data = train_source(run, verbose=verbose)
    
        if flag:
            for t_id, t_data in data:
                flags = t_data[det['hitfinder']]['data.hitFlag']
                stacked_dict = stack_agipd_dict(t_data)
                images = stacked_dict['image.data']
                pulseIDs = stacked_dict['image.pulseId']
                print(t_id, flush=True)
                for flag, p_id, image in zip(flags, pulseIDs, images):
                    if flag==0: continue 
                    else: print(flag)
                    yield t_id, p_id, image
        else:
            for t_id, t_data in data:
                stacked_dict = stack_agipd_dict(t_data)
                images = stacked_dict['image.data']
                pulseIDs = stacked_dict['image.pulseId']
                for p_id, image in zip(pulseIDs, images):
                    yield t_id, p_id, image
    else:
        ds = data_source(run)
        elements = []
        for key in det_image.keys():
            elements.append(det_image[key])
        sel = ds.select(elements, require_all=True)
        sel_flag = ds.select([det['hitfinder']], require_all=True)


        if flag:
            for t_id in train_list:
                t_id, flag_data = sel_flag.train_from_id(t_id)
                flags = flag_data[det['hitfinder']]['data.hitFlag']
                if len(np.where(flags==1)[0])==0: continue
                    
                t_id, t_data = sel.train_from_id(t_id)
                stacked_dict = stack_agipd_dict(t_data)
                images = stacked_dict['image.data']
                pulseIDs = stacked_dict['image.pulseId']
                for flag, p_id, image in zip(flags, pulseIDs, images):
                    if flag==0: continue
                    yield t_id, p_id, image
        else:
            for t_id in train_list:
                t_id, t_data = sel.train_from_id(t_id)
                stacked_dict = stack_agipd_dict(t_data)
                images = stacked_dict['image.data']
                pulseIDs = stacked_dict['image.pulseId']
                for p_id, image in zip(pulseIDs, images):
                    yield t_id, p_id, image

def getImage(run, t_id, p_id):
    '''
    Parameters
    ----------
    run : int
        The run number.
    t_id : int
        The train number.
    p_id : int
        The pulse number.

    Returns
    -------
    The image which corresponds to the run, train and pulse number
    '''
    ds = ex.open_run(proposal=proposal, run=run)
    sel = ds.select(det['agipd'])
    t_id, t_data = sel.train_from_id(t_id)
    data = stack_agipd_dict(t_data)
    images = data['image.data']
    return images[p_id]

#---------------------------------------------------------------------------------------------------------------
####################
# Spectra delivery #
####################

def spec_source(run):
    '''
    Yields
    ------
    The trainId and the hirex spectra corresponding to the trainId.
    '''
    ds = data_source(run)
    try:
        sel = ds.select(det['hirex'], require_all=True)
    except:
        print('Hirex did not record data!')
        return
    
    for t_id, t_data in sel.trains(require_all=True):
        yield t_id, t_data
    
#---------------------------------------------------------------------------------------------------------------
#########################
# Fast trainwise access #
#########################

def getPhotonEnergy(run):
    '''
    Returns
    -------
    The photon energy of a run.
    '''
    ds = data_source(run)
    sel = ds.select(det['undulator_e'])
    energy = sel[det['undulator_e'], 'actualPosition'].drop_empty_trains()[0].ndarray()[0] * 1e3 + e_offset
    return int(np.round(energy, 0))

def getPulseEnergy(run, xgm='xgm9'):
    '''
    returns 
    ----------
    a xarray with the pulse energy for the 
    to access one trainId t_id: data.sel(trainId=t_id)
    '''
    data = ex.open_run(proposal, run)
    xgm_field = det[xgm]
    intensity = data[xgm_field, 'data.intensitySa1TD'].xarray()
    filtered_intensity = intensity.where(intensity != 1).dropna(dim='dim_0').isel(dim_0=slice(1,None))
    return filtered_intensity

def getPulseEnergy_trainwise(run, xgm='xgm9', flags=True, trainList=None):
    '''
    returns 
    ----------
    a generator that gives the pulse_energy for trains using pulse_energy(run, xgm='xgm9') and data.sel(trainId=t_id).
    Only exist as an option to directly apply the hitfinderflag when flags=True
    '''
    ds = data_source(run)
    if flags:
        flag_array=ds[det['hitfinder'], 'data.hitFlag'].xarray()
    pulse_energies=getPulseEnergy(run, xgm=xgm)
    for t_id in pulse_energies.coords['trainId'].values if trainList is None else trainList:
        pulse_energies_train=pulse_energies.sel(trainId=t_id).copy()
        if flags:
            mask = flag_array.sel(trainId=t_id)
            mask=mask.rename({'trainId':'dim_0'})
            yield pulse_energies_train.where(mask).dropna(dim='dim_0')
        else:
            yield pulse_energies_train

def getInjectorPos_trainwise(run, axis='z'):
    '''
    Parameters
    ----------
    run : int
        The run number
    axis : str, optional
        The motor axis. Default is z.

    Returns
    -------
    A DataFrame that includes the injector positions of the given axis for all trains of the run.
    '''
    ds = data_source(run)
    sel = ds.select([(det['inj_{}'.format(axis)], 'encoderPosition.value')], require_all=True)
    df = sel.get_dataframe()
    df = df.rename(columns={df.columns[0]: 'injector_pos'})
    df = df.reset_index()
    return df

def getPhotonEnergy_trainwise(run):
    '''
    Parameter
    ---------
    run : int
        The run number
    
    Returns
    -------
    A DataFrame that includes the photon energy for all trains of the run.
    '''
    ds = data_source(run)
    sel = ds.select([(det['undulator_e'], 'actualPosition.value')], require_all=True)
    df = sel.get_dataframe()
    df = df.rename(columns={df.columns[0]: 'photon_energy'}) 
    df['photon_energy'] = df['photon_energy'] * 1e3 + e_offset
    df['photon_energy'] = df['photon_energy'].round().astype(int)
    df = df.reset_index()
    return df

def getTransmission_trainwise(run):
    '''
    Parameter
    ---------
    run : int
        The run number
    
    Returns
    -------
    A DataFrame that includes the photon energy for all trains of the run.
    '''
    ds = data_source(run)
    sel = ds.select([(det['att_xgm2'], 'actual.transmission.value'),
                     (det['att_xgm9'], 'actual.transmission.value')],require_all=True)
    df = sel.get_dataframe()
    df = df.rename(columns={'SPB_XTD9_ATT/MDL/MAIN/actual.transmission': 'xgm9_transmission', 
                            'SA1_XTD2_ATT/MDL/MAIN/actual.transmission': 'xgm2_transmission'}) 
    df['total_transmission'] = df['xgm9_transmission'] * df['xgm2_transmission']
    df = df.reset_index()
    return df

def getFlags(run):
    '''
    Returns
    -------
    The flags of the run.
    '''
    ds = data_source(run)
    sel = ds.select([det['hitfinder']], require_all=True)
    df = sel.get_dataframe(fields=[(det['hitfinder'], 'data.hitscore'), 
                                   (det['hitfinder'], 'data.hitFlag'), 
                                   (det['hitfinder'], 'data.pulseId')])
    df = df.rename(columns={df.columns[0]: 'pulseId', 
                            df.columns[1]: 'hitscore', 
                            df.columns[2]: 'flags'})
    df = df.reset_index()
    return df

def pulse_source_trainwise_modulewise(run, modules=[], trainlist=None):
    ds = data_source(run)
    df = getFlags(run)
    sel = ds.select(['SPB_DET_AGIPD1M-1/CORR/{}CH0:output'.format(i) for i in modules], require_all=True)
    for train in trainlist: 
        t_id, t_data = sel.train_by_id(train)
        flags = df[df['trainId']==t_id]['flags'].to_numpy()
        tmp = [t_data['SPB_DET_AGIPD1M-1/CORR/{}CH0:output'.format(i)]['image.data'] for i in modules]
        tmp = np.asarray(tmp).transpose(1, 0, 2, 3)
        tmp = np.array([tmp[i,:,:,:] for i, flag in enumerate(flags) if flag == 1])
        yield t_id, tmp

def getGeometry(run):
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

    Parameter
    ---------
    run : int
        The run number
        
    Returns
    -------
    The detector geometry object which is corrected for the given run 
    '''
    geom_fn = expPath+"Shared/geom/agipd_p008039_r0014_v16.geom"
    ref_geom = AGIPD_1MGeometry.from_crystfel_geom(geom_fn)

    try:
        ds = ex.open_run(proposal, run)
        motors = AGIPD1MQuadrantMotors(ds)
        tracker = AGIPD_1MMotors(ref_geom)
        agipd_geom = tracker.geom_at(motors.most_frequent_positions())
        return agipd_geom
    except:
        print('geometry not implemented, falling back to reference geometry')
        return ref_geom
    
#---------------------------------------------------------------------------------------------------------------
####################
# Helper functions #
####################

def stack_agipd_dict(train_data: dict):
    '''
    Parameter
    ---------
    train_data : dict
        Dictionary of the train data
        
    Returns
    -------
    dict containing stacked image of the AGIPD detector ('image.data'), pulse id ('image.pulseId'), train id (image.trainId')
    '''
    temp_dct={}
    for i in range(16):
        temp_dct['SPB_DET_AGIPD1M-1/CORR/{}CH0:output'.format(i)]=train_data['SPB_DET_AGIPD1M-1/CORR/{}CH0:output'.format(i)]
    stacked_imgs_agipd=ex.stack_detector_data(train=temp_dct,
                                              data='image.data',
                                              pattern='SPB_DET_AGIPD1M-1/CORR/(\d+)CH0:output')
    pulseIds = train_data['SPB_DET_AGIPD1M-1/CORR/0CH0:output']['image.pulseId']
    trainId = train_data['SPB_DET_AGIPD1M-1/CORR/0CH0:output']['image.trainId'][0]
    ret={'image.data' :stacked_imgs_agipd , 'image.pulseId':pulseIds ,  'image.trainId':trainId }
    return ret
    
    
class NoiseCutter():
    def __init__(self, threshold=1000, fill_value=0.):
        """
        Cuts values below threshold.
        if set_to_nan, these are set to nan, othervise
        """
        self.threshold = threshold
        self.fill_value=fill_value

    def __call__(self, image):
        ret = np.copy(image)
        ret[ret<self.threshold] = self.fill_value
        return ret

class Histogrammer():
    def __init__(self, bins, range):
        """
        Calculates Histogramms with the same bins
        """
        self.bins = bins
        self.range = range

    def __call__(self,data):
        return np.histogram(data, bins=self.bins, range=self.range)[0]

    def edges(self):
        return np.histogram_bin_edges(np.zeros(0), bins=self.bins, range=self.range)

    def centers(self):
        edges = self.edges()
        centers = (edges[1:] + edges[:-1]) / 2
        return centers

def run_format(run):
    '''
    Returns
    -------
    The formated run number.
    '''
    return '{0:04d}'.format(run)

def mask_full_flour(bad=False):
    '''
    Parameter
    ---------
    bad : bool, optional
        If True only the bad pixel mask is returned.
        Otherwise (default: False) the mask for the fluorescence is returned.

    Returns
    -------
    The mask for the fluorescence or the bad pixels (shape: 16, 512, 128).

    Module layout:
        
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
    with h5py.File(expPath+'Shared/geom/mask_hvoff_20250311.h5', 'r') as f:
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
