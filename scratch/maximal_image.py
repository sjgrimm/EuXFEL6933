import sys
sys.path.append('/home/leutloff/Software/generatorpipeline/generatorpipeline/')
import accumulators as acc

import data_helper as dh
import numpy as np

def max_image(run):

    data = dh.data_source(run)
    max_image = acc.Maximum()

    for _, _, image in data:
        max_image.accumulate(image)

    return max_image.value

def max_10_images(run, threshold):
    data = dh.data_source(run)
    max_img_data = acc.CacheMaximum(length=10, key=lambda x: x[1])

    for t_id, p_id, image in data:
        lit = np.sum(image>threshold)
        max_img_data.accumulate([image, lit])

    return max_img_data.value