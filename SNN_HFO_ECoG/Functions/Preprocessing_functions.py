#from brian2 import *
import scipy as sc
import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter,  filtfilt

#========================================================================================
# Butterworth filter coefficients
#========================================================================================   
'''
These functions are used to generate the coefficients for lowpass, highpass and bandpass
filtering for Butterworth filters.

:cutOff (int): either the lowpass or highpass cutoff frequency
:lowcut, highcut (int): cutoff frequencies for the bandpass filter
:fs (float): sampling frequency of the wideband signal
:order (int): filter order 
:return b, a (float): filtering coefficients that will be applied on the wideband signal

'''
#def butter_lowpass(cutOff, fs, order=5):
#    nyq = 0.5 * fs
#    #normalCutoff = cutOff / nyq
#    normalCutoff = 2 * cutOff / fs
#    b, a = butter(order, normalCutoff, btype='low', analog = True)
#    return b, a

def butter_lowpass(cutOff, fs, order=5):
    normalCutoff = 2 * cutOff / fs
    b, a = butter(order, normalCutoff)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#========================================================================================
# Butterworth filters
#========================================================================================   
'''
These functions apply the filtering coefficients calculated above to the wideband signal.

:data (array): vector with the amplitude values of the wideband signal 
:cutOff (int): either the lowpass or highpass cutoff frequency
:lowcut, highcut (int): cutoff frequencies for the bandpass filter
:fs (float): sampling frequency of the wideband signal
:order (int): filter order 
:return y (array): vector with amplitude of the filtered signal

'''
def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#========================================================================================
# Threshold calculation based on the noise floor
#========================================================================================   
'''
This functions retuns the mean threshold for your signal, based on the calculated 
mean noise floor and a user-specified scaling facotr that depeneds on the type of signal,
characteristics of patterns, etc.

:signal (array): amplitude of the signal
:time (array): time vector
:window (float): time window [same units as time vector] where the maximum amplitude of the signal 
                 will be calculated
:chosen_samples (int): from the maximum values in each window time, only these number of
                       samples will be used to calculate the mean maximum amplitude.
: scaling_factr (float): a percentage of the calculated threshold

'''

def find_thresholds(signal, time, window, step_size, chosen_samples, scaling_factor ):
    window_size = window
    trial_duration =np.max(time)
    num_timesteps = int(np.ceil(trial_duration / step_size))
    max_min_amplitude = np.zeros((num_timesteps, 2))
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=trial_duration,step=step_size)):        
        interval=[interval_start, interval_start + window_size]
        start_time, end_time = interval
        index = np.where(np.logical_and(time >= start_time, time <= end_time))[0]
        max_amplitude = np.max(signal[index])
        min_amplitude = np.min(signal[index])
        max_min_amplitude[interval_nr,0] = max_amplitude
        max_min_amplitude[interval_nr,1] = min_amplitude  

    threshold_up = np.mean(np.sort(max_min_amplitude[:,0])[:chosen_samples])
    threshold_dn = np.mean(np.sort(max_min_amplitude[:,1] * -1)[:chosen_samples])
    mean_threshold = scaling_factor*(threshold_up + threshold_dn)
    
    return mean_threshold

#========================================================================================
# Signal to spike conversion with refractory period
#========================================================================================   
'''
This functions retuns two spike trains, when the signal crosses the specified threshold in 
a rising direction (UP spikes) and when it crosses the specified threshold in a falling 
direction (DOWN spikes)

:time (array): time vector
:amplitude (array): amplitude of the signal
:interpfact (int): upsampling factor, new sampling frequency
:thr_up (float): threshold crossing in a rising direction
:thr_dn (float): threshold crossing in a falling direction
:refractory_period (float): period in which no spike will be generated [same units as time vector]
'''

def signal_to_spike_refractory(interpfact, time, amplitude, thr_up, thr_dn,refractory_period):
    actual_dc = 0 
    spike_up = []
    spike_dn = []

    f = sc.interpolate.interp1d(time, amplitude)                
    rangeint = np.round((np.max(time) - np.min(time))*interpfact)
    xnew = np.linspace(np.min(time), np.max(time), num=int(rangeint), endpoint=True)                
    data = np.reshape([xnew, f(xnew)], (2, len(xnew))).T
    
    i = 0
    while i < (len(data)):
        if( (actual_dc + thr_up) < data[i,1]):
            spike_up.append(data[i,0] )  #spike up
            actual_dc = data[i,1]        # update current dc value
            i += int(refractory_period * interpfact)
        elif( (actual_dc - thr_dn) > data[i,1]):
            spike_dn.append(data[i,0] )  #spike dn
            actual_dc = data[i,1]       # update curre
            i += int(refractory_period * interpfact)
        else:
            i += 1

    return spike_up, spike_dn

    