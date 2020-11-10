import numpy as np
import copy
import random as rdm
from SNN_HFO_Ecog.Functions.Preprocessing_functions import *
from SNN_HFO_Ecog.Functions.Spike_manager_functions import *


def pre_process_data(raw_signal,time_vector, sampling_frequency, interpfact, refractory, FR_scaling_factor):

    Ecog_signal_filtered = butter_bandpass_filter(data = raw_signal,
                                                  lowcut = 250,
                                                  highcut = 500,
                                                  fs = sampling_frequency,
                                                  order=2) 

    Spiking_threshold = find_thresholds(signal = Ecog_signal_filtered, 
                                          time = time_vector, 
                                          window = 0.5, # Second
                                          step_size = 0.5, 
                                          chosen_samples = 50,
                                          scaling_factor = FR_scaling_factor)

    FR_up, FR_dn  = signal_to_spike_refractory(interpfact = interpfact, 
                                           time = time_vector,
                                           amplitude = Ecog_signal_filtered ,
                                           thr_up = Spiking_threshold, thr_dn = Spiking_threshold, 
                                           refractory_period = refractory)

    FR_up = np.asarray(FR_up)
    FR_dn = np.asarray(FR_dn)


    Signal = {}
    Signal['Ecog'] = Ecog_signal_filtered
    Signal['time'] = time_vector
    

    Spikes = {}
    Spikes['threshold'] = np.asarray(Spiking_threshold)
    Spikes['up'] = FR_up
    Spikes['dn'] = FR_dn

    return Signal, Spikes


def pre_process_data_RFR(raw_signal,time_vector, sampling_frequency,
 interpfact, refractory, FR_scaling_factor, R_scaling_factor):

    Ecog_signal_filtered_FR = butter_bandpass_filter(data = raw_signal,
                                                  lowcut = 250,
                                                  highcut = 500,
                                                  fs = sampling_frequency,
                                                  order=2) 

    Spiking_threshold_FR = find_thresholds(signal = Ecog_signal_filtered_FR, 
                                          time = time_vector, 
                                          window = 1, # Second 
                                          chosen_samples = 50,
                                          scaling_factor = FR_scaling_factor)

    FR_up, FR_dn  = signal_to_spike_refractory(interpfact = interpfact, 
                                           time = time_vector,
                                           amplitude = Ecog_signal_filtered_FR ,
                                           thr_up = Spiking_threshold_FR, thr_dn = Spiking_threshold_FR, 
                                           refractory_period = refractory)

    Ecog_signal_filtered_R = butter_bandpass_filter(data = raw_signal,
                                                  lowcut = 80,
                                                  highcut = 250,
                                                  fs = sampling_frequency,
                                                  order=2) 

    Spiking_threshold_R = find_thresholds(signal = Ecog_signal_filtered_R, 
                                          time = time_vector, 
                                          window = 1, # Second 
                                          chosen_samples = 50,
                                          scaling_factor = R_scaling_factor)

    R_up, R_dn  = signal_to_spike_refractory(interpfact = interpfact, 
                                           time = time_vector,
                                           amplitude = Ecog_signal_filtered_R ,
                                           thr_up = Spiking_threshold_R, thr_dn = Spiking_threshold_R, 
                                           refractory_period = refractory)
    R_up = np.asarray(R_up)
    R_dn = np.asarray(R_dn)

    FR_up = np.asarray(FR_up)
    FR_dn = np.asarray(FR_dn)


    Signal = {}
    Signal['Ecog_R'] = Ecog_signal_filtered_R
    Signal['Ecog_FR'] = Ecog_signal_filtered_FR
    Signal['time'] = time_vector
    

    Spikes = {}
    Spikes['threshold_R'] = np.asarray(Spiking_threshold_R)
    Spikes['threshold_FR'] = np.asarray(Spiking_threshold_FR)
    Spikes['R_up'] = R_up
    Spikes['R_dn'] = R_dn
    Spikes['FR_up'] = FR_up
    Spikes['FR_dn'] = FR_dn

    return Signal, Spikes


