#from brian2 import *
import numpy as np
import scipy.io as sio
import scipy as sc

'''
This set of functions consist of a function that wraps all the spikes from different channels
into vector of spike times and neuron ID (similar to the vectors returned by DYNAPSE)
and a mean firing rate giving a vector of spiketimes and neuron ID with configurable
window and step size for computing the average activity.

'''


def concatenate_spikes(spikes_list):
    '''
    Get spikes per channel in a dictionary and concatenate them in one ingle vector with 
    spike times and neuron ids.
    :param spikes_list (dict): dict where the key is the channel name and contains a vector
                               with spike times 
    :return all_spiketimes (array): vector of all spike times
    :return all_neuron_ids (array): vector of all neuron ids 
    '''

    all_spiketimes = []
    all_neuron_ids = []
    channel_nr = 0
    for key in spikes_list:
        if channel_nr == 0:
            all_spiketimes = spikes_list['%s' % key]
            all_neuron_ids = np.ones_like(all_spiketimes) * channel_nr
            channel_nr += 1
        else:
            new_spiketimes = spikes_list['%s' % key]
            all_spiketimes = np.concatenate(
                (all_spiketimes, new_spiketimes), axis=0)
            all_neuron_ids = np.concatenate((all_neuron_ids,
                                             np.ones_like(new_spiketimes) * channel_nr), axis=0)
            channel_nr += 1

    sorted_index = np.argsort(all_spiketimes)
    all_spiketimes_new = all_spiketimes[sorted_index]
    all_neuron_ids_new = all_neuron_ids[sorted_index]
    return all_spiketimes_new, all_neuron_ids_new


def get_meanfiringrate_from_network_activity(time_stamps, neuron_ids, num_neurons, trial_duration,
                                             step_size, window_size):
    '''
    Calculate mean firing rate from network activity for all neurons over entire trial duration.
    :param time_stamps (array):     spike times
    :param neuron_ids (array):      neuron ids
    :param num_neurons (int):       number of neurons in network
    :param trial_duration (float):  duration of one trial in seconds
    :param step_size (float):       time in seconds by which window to calculate mfr is shifted per step
    :param window_size (float):     size of timebin to calculate mfr
    :return mfr (array):            calculated mean firing rate in each specified time window
    '''

    assert step_size <= window_size, "Window size is smaller than step size. Please assure: step_size<=window_size"

    # to get same number of time steps for all trials indep of spiking behaviour
    num_timesteps = int(np.ceil(trial_duration / step_size))

    # calc mfr for every trial and write to trial.mfr

    mfr = np.zeros([num_neurons, num_timesteps])
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=trial_duration, step=step_size)):

        interval = [interval_start, interval_start + window_size]
        start_time, end_time = interval
        index = np.where(np.logical_and(
            time_stamps >= start_time, time_stamps <= end_time))[0]
        spike_count = np.bincount(neuron_ids.astype(
            int)[index], minlength=num_neurons)
        interval_duration = end_time - start_time
        a = np.asarray(spike_count / interval_duration)
        mfr[:, interval_nr] = a

    return mfr
