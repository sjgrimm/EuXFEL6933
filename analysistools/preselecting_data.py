import sys
sys.path.append('/gpfs/exfel/exp/SPB/202501/p006933/usr/Software/analysistools')
import data_helper as dh
import focus_scan as fs
import numpy as np
import pandas as pd
# []
# {}
import extra_data as ex

class Run():
    def __init__(self, run_number: int, proposal: int=6933):
        self.run_number = run_number
        self.proposal = proposal
        self.data_source = dh.data_source(self.run_number)
        self.train_data = self.getTrainData()
        self.pulse_data = self.getPulseData()
        self.reduced_pulse_data = self.getReducedPulseData()              #flag==1
        self.bad_pulse_data = self.getBadPulseData()                      #flag==0
        self.all_data = self.getAllData()
        self.reduced_data = self.getReducedData()                         #flag==1
        self.bad_data = self.getBadData()                                 #flag==0
        self.hitrate = len(self.reduced_pulse_data)/len(self.pulse_data)
        self.geom = dh.getGeometry(self.run_number)

    def getTrainData(self):
        '''
        Returns
        -------
        All 1d data which is only recorded for whole trains as pd.DataFram.
        They are sorted by trainId.
        '''
        sel = self.data_source.select([(dh.det['undulator_e'], 'actualPosition.value'),
                                       (dh.det['inj_x'], 'encoderPosition.value'), 
                                       (dh.det['inj_y'], 'encoderPosition.value'),
                                       (dh.det['inj_z'], 'encoderPosition.value'),
                                       (dh.det['agipd_z'], 'encoderPosition.value'),
                                       (dh.det['att_xgm2'], 'actual.transmission.value'),
                                       (dh.det['att_xgm9'], 'actual.transmission.value')], 
                                      require_all=True)
        
        df = sel.get_dataframe()
        df = df.rename(columns={dh.det['undulator_e']+'/actualPosition':'photon_energy', 
                                dh.det['inj_x']+'/encoderPosition':'inj_pos_x', 
                                dh.det['inj_y']+'/encoderPosition':'inj_pos_y', 
                                dh.det['inj_z']+'/encoderPosition':'inj_pos_z', 
                                dh.det['agipd_z']+'/encoderPosition':'agipd_pos_z',
                                dh.det['att_xgm2']+'/actual.transmission':'xgm2_transmission',
                                dh.det['att_xgm9']+'/actual.transmission':'xgm9_transmission'})
        
        df['photon_energy'] = df['photon_energy'] * 1e3 + dh.e_offset
        df['photon_energy'] = df['photon_energy'].round().astype(int)
        df['total_transmission'] = df['xgm9_transmission'] * df['xgm2_transmission']
        column_order = ['inj_pos_x', 'inj_pos_y', 'inj_pos_z', 'agipd_pos_z', 'photon_energy', 
                        'xgm2_transmission', 'xgm9_transmission', 'total_transmission']
        df = df[column_order]
        df = df.reset_index()
        return df

    def getPulseData(self):
        '''
        Returns
        -------
        All 2d data which is recorded per pulse as pd.DataFrame.
        They are sorted by trainId and pulseId.
        '''
        sel_flags = self.data_source.select([(dh.det['hitfinder'], 'data.pulseId'),
                                             (dh.det['hitfinder'], 'data.hitFlag'), 
                                             (dh.det['hitfinder'], 'data.hitscore')], 
                                            require_all=True)
        df_flags = sel_flags.get_dataframe()
        df_flags = df_flags.rename(columns={dh.det['hitfinder']+'/data.pulseId': 'pulseId', 
                                            dh.det['hitfinder']+'/data.hitscore': 'hitscore', 
                                            dh.det['hitfinder']+'/data.hitFlag': 'flags'})
        df_flags = df_flags.reset_index()
        df_flags = df_flags.sort_values(by=['trainId', 'pulseId'])

        #xgm2 = self.getPulseEnergy(xgm='xgm2').stack(z=('trainId', 'dim_0'), create_index=False).values
        #xgm9 = self.getPulseEnergy(xgm='xgm9').stack(z=('trainId', 'dim_0'), create_index=False).values

        # xgm2 = self.getPulseEnergy(xgm='xgm2')
        # xgm9 = self.getPulseEnergy(xgm='xgm9')
        # ix = np.arange(1, 352)*4+20
        # hirex = self.data_source[dh.det['hirex'], 'data.frameNumber'].ndarray()[:, ix].reshape(-1)
        
        # df_flags['pulse_energy_xgm2'] = xgm2
        # df_flags['pulse_energy_xgm9'] = xgm9
        # df_flags['hirex_frame_number'] = hirex

        column_order = ['trainId', 'pulseId', 'flags', 'hitscore'] 
                       # 'pulse_energy_xgm2', 'pulse_energy_xgm9', 'hirex_frame_number']
        df = df_flags[column_order]

        return df

    def getPulseEnergy(self, xgm: str='xgm9'):
        '''
        Parameter
        ---------
        xgm : str, optional
            Determines which gas detector should be used.
            Default is xgm9.

        Returns
        -------
        The pulse energy for the given gas detector as 1d ndarray (2d xarray).
        (The first dimension is the trainId and 
        the second dimension is the number of pulses (not pulseId!)).
        '''
        #intensity = self.data_source[dh.det[xgm], 'data.intensitySa1TD'].xarray()
        #filtered_intensity = intensity.where(intensity!=1).dropna(dim='dim_0').isel(dim_0=slice(1,None))
        intensity = self.data_source[dh.det[xgm], 'data.intensitySa1TD'].ndarray()[:, 1:]
        filtered_intensity = intensity[intensity != 1.0]
        print(len(filtered_intensity))
        return filtered_intensity

    def getReducedPulseData(self):
        '''
        Returns
        -------
        The pulse data for which the flag is equal to one as pd.DataFram.
        '''
        df = self.pulse_data
        filtered_df = df[df['flags'] == 1]
        return filtered_df

    def getBadPulseData(self):
        '''
        Returns
        -------
        The pulse data for which the flag is equal to zero as pd.DataFram.
        '''
        df = self.pulse_data
        filtered_df = df[df['flags'] == 0]
        return filtered_df

    def getAllData(self):
        '''
        Returns
        -------
        The merged pulse and train data as pd.DataFrame.
        '''
        df_pulse = self.pulse_data
        df_train = self.train_data
        df = pd.merge(df_train, df_pulse, on='trainId', how='inner')
        return df

    def getReducedData(self):
        '''
        Returns
        -------
        The merged reduced_pulse and train data as pd.DataFrame.
        '''
        df_reduced_pulse = self.reduced_pulse_data
        df_train = self.train_data
        df = pd.merge(df_train, df_reduced_pulse, on='trainId', how='inner')
        return df

    def getBadData(self):
        '''
        Returns
        -------
        The merged bad_pulse and train data as pd.DataFrame.
        '''
        df_bad_pulse = self.bad_pulse_data
        df_train = self.train_data
        df = pd.merge(df_train, df_bad_pulse, on='trainId', how='inner')
        return df