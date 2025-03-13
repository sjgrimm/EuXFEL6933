'''
important finding: In the hirex data only every fourth image corresponds to a spectrum. It records for the maximum rate.
First spectrum seems to be at index 20
drop first spectrum, the first agipd image in the raw data is used as a dark and dropped when moving to the calibrated version
'''
import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/')
from analysistools import data_helper as dh
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
from generatorpipeline.generatorpipeline import generatorpipeline as gp
from generatorpipeline.generatorpipeline import accumulators
trains_dh=dh.train_source(run=8)
hist=dh.Histogrammer(bins=100, range=(10,500))

def mask_spectrum(line):
    mask=np.ones(len(line), dtype=bool)
    mask[201:208]=0
    mask[1282:1284]=0
    mask[1808:1824]=0
    return mask

default_mask=mask_spectrum(np.ones(2560))

def check_for_spectrum(line, mask=default_mask):
    histogram=hist(line[mask])
    hist_mask=hist.centers()>50
    return np.sum(histogram[hist_mask])> 10

def autocorrelate(spectrum):
    spectrum2=spectrum[~np.isnan(spectrum)]
    return scipy.signal.correlate(spectrum2, spectrum2, mode="full", method='fft')


@gp.pipeline(20)
def mean_autocorrelation_train(train):
    skipped_first=False
    auto=accumulators.Mean()
    data=train[1]
    spec_data=data[dh.det['hirex']]['data.adc']
    for index in range(np.shape(spec_data)[0]):
        curr_spec=spec_data[index,:]
        if check_for_spectrum(curr_spec) and skipped_first:
            auto.accumulate(autocorrelate(curr_spec))
        elif not skipped_first and check_for_spectrum(curr_spec):
            skipped_first=True
    return auto

def mean_autocorrelation_run(run=8):
    t0=time.time()
    trains=dh.spec_source(run)
    auto_trains=mean_autocorrelation_train(trains)
    acc=accumulators.Mean()
    n=0
    for spec_auto in auto_trains:
        acc.accumulate(spec_auto)
    t1=time.time()
    print(t1-t0)
    return acc.value
    