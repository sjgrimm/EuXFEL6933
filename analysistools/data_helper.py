'''
Created on Thursday 27.02.2025

@author: Jan Niklas Leutloff
@author: Nils Hendrik Muthreich
@author: Paul Rasmus Buchin

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
    'frames': 'SPB_IRU_AGIPD1M1/REDU/LITFRM:output',             # XGM2 values with pulseId
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
    good_data_ratio = number_of_trains/original_number_of_trains
    if good_data_ratio<0.98:
        print('{0:.2%} of the data of run {1:} is used!'.format(good_data_ratio, run), flush=True)
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
    pulse_ids = data['image.pulseId']
    images = data['image.data']
    index = np.where(pulse_ids==p_id)[0][0]
    return images[index]

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

def getPulseEnergy(self, npulse_per_train, xgm: str='xgm9'):
        '''
        Parameter
        ---------
        npulse_per_train : int
            Number of pulses per train.
        xgm : str, optional
            Determines which gas detector should be used.
            Default is xgm9.

        Returns
        -------
        The pulse energy for the given gas detector as 1d ndarray.
        '''
        # The first pulse (pulseId==0) is not recorded by the xgms because there is no pulse (dark image for the agipd)
        # The dark image of the agipd is recorded but filtered out for our data analysis
        # The agipd record all in all npulses but because the first is the dark the last real pulse isn't recorded
        # So one has to cut of the last pulse of each train of the xgm and also of hirex since the agipd doesn't record it
        intensity = data_source[dh.det[xgm], 'data.intensitySa1TD'].ndarray()
        filtered_intensity = [t_intensity[t_intensity != 1.0][:npulse_per_train] for t_intensity in intensity]
        filtered_intensity = np.asarray(filtered_intensity).reshape(-1)
        return filtered_intensity

#def getPulseEnergy_trainwise(run, xgm='xgm9', flags=True, trainList=None):
#    '''
#    returns 
#    ----------
#    a generator that gives the pulse_energy for trains using pulse_energy(run, xgm='xgm9') and data.sel(trainId=t_id).
#    Only exist as an option to directly apply the hitfinderflag when flags=True
#    '''
#    ds = data_source(run)
#    if flags:
#        flag_array=ds[det['hitfinder'], 'data.hitFlag'].xarray()
#    pulse_energies=getPulseEnergy(run, xgm=xgm)
#    for t_id in pulse_energies.coords['trainId'].values if trainList is None else trainList:
#        pulse_energies_train=pulse_energies.sel(trainId=t_id).copy()
#        if flags:
#            mask = flag_array.sel(trainId=t_id)
#            mask=mask.rename({'trainId':'dim_0'})
#            yield pulse_energies_train.where(mask).dropna(dim='dim_0')
#        else:
#            yield pulse_energies_train


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
    df = df.rename(columns={det['hitfinder']+'/data.pulseId': 'pulseId', 
                            det['hitfinder']+'/data.hitscore': 'hitscore', 
                            det['hitfinder']+'/data.hitFlag': 'flags'})
    df = df.reset_index()
    df = df.sort_values(by=['trainId', 'pulseId'])
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

def mask_full_fluor(bad=False):
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

def getTileCorners(module, tile):
    '''
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
    
    Coordinates of the module edges in the assembled image: 
    (*: not all tiles have the same vertical position)
    
    7: (  88,  7) | (  88,532)     11: (   0, 546) | (   0,1071)
       -----------+-----------         ------------+------------
       ( 215,  7) | ( 215,532)         ( 127, 546) | ( 127,1071)
    
    6: ( 231,  1) | ( 231,526)     10: ( 152, 547) | ( 152,1072)
       -----------+-----------         ------------+------------
       ( 358,  1) | ( 215,526)         ( 279, 547) | ( 279,1072)
    
    5: ( 386,  0) | ( 388,525)      9: ( 308, 547) | ( 310,1072)
    *  -----------+-----------      *  ------------+------------
       ( 513,  0) | ( 515,525)         ( 435, 547) | ( 437,1072)
    
    4: ( 544,  0) | ( 545,525)      8: ( 466, 546) | ( 467,1071)
    *  -----------+-----------      *  ------------+------------
       ( 671,  0) | ( 672,525)         ( 593, 546) | ( 594,1071)
    
    3: ( 706, 16) | ( 709,541)     15: ( 626, 567) | ( 628,1092)
    *  -----------+-----------      *  ------------+------------
       ( 833, 16) | ( 836,541)         ( 753, 567) | ( 755,1092)
    
    2: ( 863, 16) | ( 866,541)     14: ( 782, 567) | ( 785,1092)
    *  -----------+-----------      *  ------------+------------
       ( 990, 16) | ( 993,541)         ( 909, 567) | ( 912,1092)
    
    1: (1020, 16) | (1021,541)     13: ( 939, 567) | ( 941,1092)
    *  -----------+-----------      *  ------------+------------
       (1147, 16) | (1148,541)         (1066, 567) | (1068,1092)
    
    0: (1178, 16) | (1178,541)     12: (1089, 563) | (1093,1088)
       -----------+-----------      *  ------------+------------
       (1305, 16) | (1305,541)         (1216, 563) | (1220,1088)

    Returns
    -------
    Always the upper left corner of the given tile in the given module.
    '''
    # tiles:                     0            1            2            3            4            5            6            7   | modules
    edges = np.array([[[1178,   16],[1178,   82],[1178,  148],[1178,  214],[1178,  280],[1178,  346],[1178,  412],[1178,  478]],# 0
                      [[1020,   16],[1020,   82],[1020,  148],[1020,  214],[1021,  280],[1021,  346],[1021,  412],[1021,  478]],# 1
                      [[ 863,   16],[ 863,   82],[ 864,  148],[ 864,  214],[ 865,  280],[ 865,  346],[ 866,  412],[ 866,  478]],# 2
                      [[ 706,   16],[ 706,   82],[ 707,  148],[ 707,  214],[ 707,  280],[ 708,  346],[ 708,  412],[ 709,  478]],# 3
                      [[ 544,    0],[ 544,   66],[ 544,  132],[ 544,  198],[ 545,  264],[ 545,  330],[ 545,  396],[ 545,  462]],# 4
                      [[ 386,    0],[ 387,   66],[ 387,  132],[ 387,  198],[ 387,  264],[ 388,  330],[ 388,  396],[ 388,  462]],# 5
                      [[ 231,    1],[ 231,   67],[ 231,  133],[ 231,  199],[ 231,  265],[ 231,  331],[ 231,  397],[ 231,  463]],# 6
                      [[  88,    7],[  88,   73],[  88,  139],[  88,  205],[  88,  271],[  88,  337],[  88,  403],[  88,  469]],# 7
                      [[ 466,  546],[ 466,  612],[ 466,  678],[ 466,  744],[ 466,  810],[ 467,  876],[ 467,  942],[ 467, 1008]],# 8
                      [[ 308,  547],[ 308,  613],[ 309,  679],[ 309,  745],[ 309,  811],[ 309,  877],[ 309,  943],[ 310, 1009]],# 9
                      [[ 152,  547],[ 152,  613],[ 152,  679],[ 152,  745],[ 152,  811],[ 152,  877],[ 152,  943],[ 152, 1009]],#10
                      [[   0,  546],[   0,  612],[   0,  678],[   0,  744],[   0,  810],[   0,  876],[   0,  942],[   0, 1008]],#11
                      [[1089,  563],[1090,  629],[1090,  695],[1091,  761],[1091,  827],[1092,  893],[1093,  959],[1093, 1025]],#12
                      [[ 939,  567],[ 939,  633],[ 940,  699],[ 940,  765],[ 940,  831],[ 941,  897],[ 941,  963],[ 941, 1029]],#13
                      [[ 782,  567],[ 783,  633],[ 783,  699],[ 784,  765],[ 784,  831],[ 784,  897],[ 785,  963],[ 785, 1029]],#14
                      [[ 626,  567],[ 627,  633],[ 627,  699],[ 627,  765],[ 627,  831],[ 628,  897],[ 628,  963],[ 628, 1029]],#15
                     ], dtype=int)
    
    return edges[module][tile]

def assembleImage(not_assembled_image):
    '''
    Parameter
    ---------
    not_assembled_image : ndarray
        The not assembled image with shape (16, 512, 128)

    Returns
    -------
    The assebled image with shape (1306, 1093)
    '''
    ret_img = np.zeros((1306, 1093))
    
    for m_nr, module in enumerate(not_assembled_image):
        
        trafo = np.rot90(module, axes=(0, 1)) if m_nr<8 else np.rot90(module, axes=(1, 0))
        
        for t_nr in range(8):
            ref_y, ref_x = getTileCorners(m_nr, t_nr)
            ret_img[ref_y:ref_y+128, ref_x:ref_x+64] = trafo[:,t_nr*64:(t_nr+1)*64]

    return ret_img

def imageCoordinate(coordinate, module):
    '''
    Parameter
    ---------
    coordinate : list(int, int)
        The list of coordinates (y, x)
    module : int
        The module number

    Returns
    -------
    The coordinate in the 2d image of the given 3d coordinates.
    '''
    [y_index, x_index] = coordinate

    if module<8:
        y_trafo = 127-x_index
        x_trafo = y_index
    else:
        y_trafo = x_index
        x_trafo = 511-y_index

    t_nr = int(x_trafo/64)
    ref_y, ref_x = getTileCorners(module, t_nr)

    new_y = ref_y + y_trafo
    new_x = ref_x + x_trafo-t_nr*64
    
    return [new_y, new_x]

# Reference image to identify the module for a given image coordinate
ones = np.ones((512, 128), dtype=int)
ref_modules = np.asarray([ones+i for i in range(16)])
ass_ref_modules = assembleImage(ref_modules)

def stackCoordinate(coordinate):
    '''
    Parameter
    ---------
    coordinate : list(int, int)
        The coordinates from the 2d image.

    Returns
    -------
    The coordinates in the 3d stack of modules.
    '''
    [y_index, x_index] = coordinate
    
    m_nr = int(ass_ref_modules[y_index][x_index]-1)
    ref_y, ref_x = getTileCorners(m_nr, 0)
    module_x_index = x_index - ref_x
    t_nr = int(module_x_index/66)# tile_width + gap_width = 64 + 2 = 66
    
    ref_y, ref_x = getTileCorners(m_nr, t_nr)
    y_index = y_index-ref_y
    x_index = (x_index-ref_x)+t_nr*64

    if m_nr<8:
        y_trafo = x_index
        x_trafo = 127-y_index
    else:
        y_trafo = 511-x_index
        x_trafo = y_index
        
    return [y_trafo, x_trafo], m_nr

def assembleStack(image):
    '''
    Parameter
    ---------
    image : ndarray
        The input image with shape (1306, 1093).

    Returns
    -------
    The stack of modules extracted from the given image with shape (16, 512, 128).
    '''
    stack = []

    for m_nr in range(16):
        module = np.zeros((128, 512))
        for t_nr in range(8):
            ref_y, ref_x = getTileCorners(m_nr, t_nr)
            module[:,t_nr*64:(t_nr+1)*64] = image[ref_y:ref_y+128, ref_x:ref_x+64]

        trafo = np.rot90(module, axes=(0, 1)) if m_nr>=8 else np.rot90(module, axes=(1, 0))
        stack.append(trafo)

    return np.asanyarray(stack)