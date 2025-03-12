'''
Created on Thursday 27.02.2025

@author: Jan Niklas Leutloff

import sys
sys.path.append('gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools')
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

expPath = '/gpfs/exfel/exp/SPB/202501/p006933/usr/'#/gpfs/exfel/u/usr/SPB/202501/p006933/Software
proposal = 6933

#---------------------------------------------------------------------------------------------------------------

f_threshold = 1

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
    #'test1': 'SPB_EHD_IBS/CAM/1:daqOutput',             # doesn't work
    #'test3': 'SPB_IRU_AGIPD1M1/REDU/LITFRM:output',     # weird
    #'test4': 'ACC_SYS_DOOCS/CTRL/BEAMCONDITIONS'        # kParameter
    #'test5': 'SPB_IRU_AGIPD1M1/MDL/DATA_SELECTOR',      # not usefull
    #'test6': 'SPB_IRU_AGIPD1M/PSC/HV',                  # not usefull
    
}

#---------------------------------------------------------------------------------------------------------------

def train_source(run, verbose=False):
    '''
    Parameter
    ---------
    run : int
        Run number
    verbose : bool, optional
        Wheather to print debug information (True) or not (default: False)
        
    Yields
    ------
    The trainId as int and the corresponding data of the train as dict
    '''
    geom = getGeometry(run)
    ds = ex.open_run(proposal=proposal, run=run)
    if verbose:    
        ds.info()
        ds.all_sources
        
    elements = []
    for key in det.keys():
        try:
            ds.select(det[key])
            elements.append(det[key])
        except: 
            print(f"{key} not detected")
    sel = ds.select(elements)#, 'image.data')
    #sel = ds.select(sorted(det.keys()), require_all=True)
    
    for t_id, t_data in sel.trains(require_all=True):
        yield t_id, t_data

def data_source(run, verbose=False):
    '''
    !!! DEPRECATED !!!
    Will be removed in a newer version.
    Make sure to update your legacy code before the next release.
    '''
    return train_source(run, verbose)
    

def pulse_source(run, flag=False, verbose=False):
    '''
    Parameter
    ---------
    run : int
        Run number
    verbose : bool, optional
        Wheather to print debug information (True) or not (default: False)
        
    Yields
    ------
    The trainId and the pulseId as int, respectively and the corresponding stacked image
    '''
    data = train_source(run, verbose=verbose)

    if flag:
        for t_id, t_data in data:
            flags = t_data[det['hitfinder']]['data.hitFlag']
            stacked_dict = stack_agipd_dict(t_data)
            images = stacked_dict['image.data']
            pulseIDs = stacked_dict['image.pulseId']
            for flag, p_id, image in zip(flags, pulseIDs, images):
                if flag==0: continue
                yield t_id, p_id, image
    else:
        for t_id, t_data in data:
            stacked_dict = stack_agipd_dict(t_data)
            images = stacked_dict['image.data']
            pulseIDs = stacked_dict['image.pulseId']
            for p_id, image in zip(pulseIDs, images):
                yield t_id, p_id, image

def getImage(run, t_id, p_id):
    ds = ex.open_run(proposal=proposal, run=run)
    elements = []
    for key in det.keys():
        elements.append(det[key])
    sel = ds.select(elements)
    return ds[det['agipd']]['image.data'].dask_array()[t_id, p_id, :, :]

def getGeometry(run):
    '''
    Parameters
    ----------
    run : int
        The run number
        
    Returns
    -------
    The detector geometry object which is corrected for the given run 
    '''
    
    geom_fn = expPath+"Shared/geom/agipd_p008039_r0014_v16.geom"
    ref_geom = AGIPD_1MGeometry.from_crystfel_geom(geom_fn)

    # does not work for the first 4 runs
    # TODO check whether it works for the next ones
    if True: #run < 5: 
        return ref_geom
    else: 
        ds = ex.open_run(proposal, run)
        motors = AGIPD1MQuadrantMotors(ds)
        tracker = AGIPD_1MMotors(ref_geom)
        agipd_geom = tracker.geom_at(motors.most_frequent_positions())
        return agipd_geom
    
    #agipd_geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=[
    #    (-525, 625),
    #    (-550, -10),
    #    (520, -160),
    #    (542.5, 475),
    #])


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
    stacked_imgs_agipd=ex.stack_detector_data(train=temp_dct, data='image.data', pattern='SPB_DET_AGIPD1M-1/CORR/(\d+)CH0:output')
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
