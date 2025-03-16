from datetime import datetime 
starttime = datetime.now()
import numpy as np
from damnit import Damnit
import extra_data as ex
import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools/')
import data_helper as dh
import argparse

parser = argparse.ArgumentParser(description='Bitte folgende Run Parameter angeben.')
parser.add_argument('--run', '-r', type=int, required=True, help='Run number')
args = parser.parse_args()
run = args.run

def main(run=run):
    data = ex.open_run(6933, run)
    ds = dh.data_source(run)
    g = dh.pulse_source(run) 
    img_sum = np.zeros((16,512,128))
    N=0
    try:
        while True:
            _, _, image = next(g)
            N += 1
            img_sum += image
    except StopIteration:
        pass
        
    average = img_sum / N
    np.save(f'/gpfs/exfel/u/usr/SPB/202501/p006933/Results/RunAverages/whole_r{run}_average.npy', average)
    print(f'Averaging the whole run {run} took {datetime.now()-starttime}.')
    return

if __name__ == '__main__':
    main()