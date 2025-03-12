'''
Created on Friday 28.02.2025

@author: Jan Niklas Leutloff
'''

import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

def radialProfile(img, center=None):
    '''
    Parameter
    ---------
    img : np.ndarray
        The assembled input image
    center : (int, int) optional
        (y-coordinate, x-coordinate) of the center
        The center of the input image or the center of
        the debye rings which are displayed in the image

    Returns
    -------
    The radial profile of the image
    '''
    if center is None:
        #        y-coordinate        x-coordinate
        center = img.shape[0] // 2,  img.shape[1] // 2
        
    y, x = np.indices((img.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    tbin = np.bincount(r.ravel(), img.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    return radialprofile

def radialProfile_pyfai(not_assembled_image, geom, dist=None, wavelength=None):
    '''
    Parameter
    ---------
    not_assembled_image : np.ndarray
        The not assembled input image
    geom : extra_geom.detectors.AGIPD_1MGeometry
        The geometry of the used detector
    dist : float, optional
        The sample-detector distance (m)
    wavelength : float
        The wavelength (m)

    Returns
    -------
    The radial axis in mm and the corresponding radial profile of the image
    '''
    ai = AzimuthalIntegrator(detector=geom.to_pyfai_detector(),
                             dist=dist,  # sample-detector distance (m)
                             wavelength=wavelength#(12.3984 / 9.3) * 1e-10  # wavelength (m)
                            )
    
    rint, I = ai.integrate1d(not_assembled_image.reshape(16*512, 128),
                             npt=300,
                             unit="r_mm"
                            )
    
    return rint, I