import numpy as np


# TO DO: add description of the function

def detect_HFO(trial_duration, spike_monitor, neuron_spike_monitor, step_size, window_size):

    periods_of_HFO = np.array([[0, 0]])
    # ==============================================================================
    # Detect HFO
    # ==============================================================================
    assert step_size <= window_size
    # to get same number of time steps for all trials independently of spiking behaviour
    num_timesteps = int(np.ceil(trial_duration / step_size))

    mfr = np.zeros([num_timesteps])
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=trial_duration, step=step_size)):
        interval = [interval_start, interval_start + window_size]
        start_time, end_time = interval
        index = np.where(np.logical_and(
            spike_monitor >= start_time, spike_monitor <= end_time))[0]
        interval_duration = end_time - start_time
        a = np.asarray(index.size / interval_duration)
        mfr[interval_nr] = a
        # if index.size  != 0:
        #    periods_of_HFO = np.concatenate((periods_of_HFO,np.array([[start_time,end_time]])))

    mfr_ones = np.where(mfr != 0)
    mfr_binary = np.zeros(mfr.size)
    mfr_binary[mfr_ones] = 1

    signal_rise = []
    signal_fall = []

    binary_signal = mfr_binary

    for i in range(binary_signal.size-1):
        if i == 0 and binary_signal[0] == 1:
            signal_rise.append(i)
        if i > 0 and binary_signal[i] == 1 and binary_signal[i-1] == 0:
            signal_rise.append(i)
        elif binary_signal[i] == 1 and binary_signal[i+1] == 0:
            signal_fall.append(i)
        if i == binary_signal.size-2 and binary_signal[i] == 1:
            signal_fall.append(i)

    signal_rise = np.asarray(signal_rise)
    signal_fall = np.asarray(signal_fall)

    HFO_identificaiton_time = np.arange(
        start=0, stop=trial_duration, step=step_size)
    HFO_identificaiton_signal = np.zeros(HFO_identificaiton_time.size)

    for i in range(signal_rise.size):
        HFO_identificaiton_signal[signal_rise[i]:signal_fall[i]] = 1

    identified_HFO = signal_rise.size

    if signal_rise.size != 0:
        start_period_HFO = HFO_identificaiton_time[signal_rise]
        stop_period_HFO = HFO_identificaiton_time[signal_fall]
        periods_of_HFO = np.array([start_period_HFO, stop_period_HFO])
    else:
        periods_of_HFO = np.array([0, 0])

    HFO_detection = {}
    HFO_detection['total_HFO'] = identified_HFO
    HFO_detection['time'] = HFO_identificaiton_time
    HFO_detection['signal'] = HFO_identificaiton_signal
    HFO_detection['periods_HFO'] = periods_of_HFO

    return HFO_detection
