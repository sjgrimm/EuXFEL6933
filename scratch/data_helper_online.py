'''
Created on Thursday 27.02.2025

@author: Jan Niklas Leutloff

import sys
sys.path.append('/sdf/data/lcls/ds/cxi/cxily3123/results/jleutloff/Analysis/helper/')
import data_helper_online as dho
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

import data_helper as dh

import extra_data as ex
from extra_data.components import AGIPD1M
from extra_geom import AGIPD_1MGeometry

from karabo_bridge import Client

#---------------------------------------------------------------------------------------------------------------

expPath = 'GPFS/exfel/exp/SPB/202501/p006933/usr/Shared/'
proposal = 6933

#---------------------------------------------------------------------------------------------------------------

det = {
    'hirex': 'SA1_XTD9_HIREX/CORR/GOTTHARD_RECEIVER:daqOutput',  # Spectrometer
    'xgm2': 'SA1_XTD2_XGM/XGM/DOOCS:output',                     # Gas detector in SASE1
    'xgm9': 'SPB_XTD9_XGM/XGM/DOOCS:output',                     # Gas detector next to SPB
    'agipd': 'SPB_DET_AGIPD1M-1/CORR/*',                         # Main 2d detector
    'agipd_z': 'SPB_IRU_AGIPD1M/MOTOR/Z_STEPPER',                # z position of AGIPD
    'sample_x': 'SPB_IRU_INJMOV/MOTOR/X',                        # positions of the injector
    'sample_y': 'SPB_IRU_INJMOV/MOTOR/Y',                        #
    'sample_z': 'SPB_IRU_INJMOV/MOTOR/Z',                        #
    'undulator_e': 'SPB_XTD2_UND/DOOCS/ENERGY',                  # Energy set by the undulator
    'attenuators': 'SPB_XTD9_ATT/MDL/MAIN'                       # Attenuation
}

#---------------------------------------------------------------------------------------------------------------

def serve_trains(host, sock='REQ'):
    '''
    Generator for the online data stream.
    Input: 
        host: ip address of data stream
              something like tcp://max-exfl261.desy.de:1234
        type: ???
    Output:
        dictionary of values for current event
    '''
    # Generate a client to serve the data
    c = Client(endpoint=host, sock=sock)

    # Return the newest event in the datastream using an iterator construct
    for ret in c:
        try:
            ret[0]['SPB_DET_AGIPD1M-1/CORR/stacked:output']=dh.stack_agipd_dict(ret[0])
        except KeyError:
            print('stacking agipd tiles failed')
        yield {'data':ret[0], 'meta':ret[1]} #it comes out as a dict so the we have a consistent datastream


def serve_pulses(host, sock='REQ'):
    
    trains = serve_trains(endpoint=host, sock=sock)

    for train in trains:
        data = train['data']
        meta = train['meta']
        images = data['image.data'].transpose(3, 0, 2, 1)
        trainID = data['image.trainId']
        pulseID = data['image.pulseId']

        for t_id, p_id, image in zip(trainID, pulseID, images):
            yield t_id, p_id, image