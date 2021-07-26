from brian2 import *

import os
import scipy
import scipy.io as sio
import seaborn as sb

import random as rdm

# IMPORT FUNCTIONS
from SNN_HFO_ECoG.Functions.Ecog_set_functions import *
from SNN_HFO_ECoG.Functions.Dynapse_biases_functions import *
from SNN_HFO_ECoG.Functions.HFO_detection_functions import *

# IMPORT  Teili functions
from teili.tools.plotter2d import Plotter2d
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

# ===============================================================================================================
# Paths
# ===============================================================================================================
#root_path = './'
root_path = '/Users/karla/DataFiles/'
new_data_path = root_path + 'ECoG_Data/data_python/'

root_path_save = './'
save_path_figures = root_path_save + 'Figures/'
save_path_results = root_path_save + 'ArtifactRejection/HFOIdentificationResults/'

# ===============================================================================================================
# Set general Network parameters
# ===============================================================================================================
# extra time brian simulation will run after last signal time in seconds
extra_simulation_time = 0.050
refractory = 3e-4  # in miliseconds

# ==============================================================================
# Network parameters: Number of neurons
# ==============================================================================
Input_channels = 2
Hidden_neurons = 256

# ==============================================================================
# Network parameters: Time constant and weight ranges
# ==============================================================================
# The minimum time constant of the excitatory synapse is 3 ms and maximum 6 ms
# the corresponding inhibitory time constant is exc_tau - a factor that goes from 0.3-1 ms

min_time_constant = 3
max_time_constant = 6
min_subtraction_time_constant = 0.3
max_subtraction_time_constant = 1

#min_subtraction_time_constant = 0.4
#max_subtraction_time_constant = 1.2

# We will have only 2 possible excitatory weights, the inhibitory weight will be the same
possible_weights_exc = [1000, 2000]
# ==============================================================================
# Network parameters: Time constants and weights distributions
#==============================================================================
# Up and down channels must have opposite effect on postsynaptic neuron 
# (neurons in the hidden layer). To get more homogeneity Up (Dn) channels excite (inhibit)
# half of the hidden neurons and inhibit (excite) the other half.

# ====================================
# Time constants for Fast Ripple ch
# ====================================
#min_time_constant = 2
#max_time_constant = 3

input_hidden_time_constants_FR_up_exc = numpy.random.uniform(
    low=min_time_constant, high=max_time_constant, size=int(Hidden_neurons/2))
input_hidden_time_constants_FR_dn_inh = input_hidden_time_constants_FR_up_exc-(numpy.random.uniform(low=min_subtraction_time_constant,
                                                                                                    high=max_subtraction_time_constant,
                                                                                                    size=int(Hidden_neurons/2)))
input_hidden_time_constants_FR_dn_exc = numpy.random.uniform(
    low=min_time_constant, high=max_time_constant, size=int(Hidden_neurons/2))
input_hidden_time_constants_FR_up_inh = input_hidden_time_constants_FR_dn_exc-(numpy.random.uniform(low=min_subtraction_time_constant,
                                                                                                    high=max_subtraction_time_constant,
                                                                                                    size=int(Hidden_neurons/2)))

input_hidden_time_constants_FR_up = np.concatenate(
    (input_hidden_time_constants_FR_up_exc, input_hidden_time_constants_FR_up_inh))
input_hidden_time_constants_FR_dn = np.concatenate(
    (input_hidden_time_constants_FR_dn_inh, input_hidden_time_constants_FR_dn_exc))

# ====================================
# Get all time constants in a vector
# ====================================
input_hidden_time_constants = np.concatenate(
    (input_hidden_time_constants_FR_up, input_hidden_time_constants_FR_dn))


# ====================================
# Weights for Fast Ripple channels
# ====================================
input_hidden_weights_FR_up_exc = rdm.choices(
    possible_weights_exc, k=int(Hidden_neurons/2))
input_hidden_weights_FR_dn_inh = [-1*x for x in input_hidden_weights_FR_up_exc]
input_hidden_weights_FR_dn_exc = rdm.choices(
    possible_weights_exc, k=int(Hidden_neurons/2))
input_hidden_weights_FR_up_inh = [-1*x for x in input_hidden_weights_FR_dn_exc]

# ====================================
# Get all weights in a vector
# ====================================
input_hidden_weights = input_hidden_weights_FR_up_exc +\
    input_hidden_weights_FR_up_inh +\
    input_hidden_weights_FR_dn_inh +\
    input_hidden_weights_FR_dn_exc


# ========================================================================================
# General parameters
# ========================================================================================
sampling_frequency = 2000
FR_scaling_factor = 0.5
interpfact = 35000
refractory = 3e-4
recording_type = 'pre'

# ========================================================================================
# Loop over Patients
# ========================================================================================
list_patients = np.array([1, 2, 3, 4, 5, 6, 7, 8])

for cp, current_patient in enumerate(list_patients):

    Data = sio.loadmat(new_data_path +
                       'P%i/Data_%s_Patient_%02d.mat' % (current_patient, recording_type, current_patient))

    if recording_type == 'pre':
        recording_prefix = 'Pre'
    elif recording_type == 'post':
        recording_prefix = 'Post'

    number_of_channels = Data['%s_Ecog_signal' % recording_prefix].shape[0]

    # Create dictionary to save results
    Test_Results = {}
    Test_Results['Info'] = {}
    Test_Results['Info']['Patient'] = np.array(['P%i' % current_patient])
    Test_Results['Info']['Recordings'] = np.array(
        ['%s-recorded data' % recording_type])
    Test_Results['Info']['Channels'] = np.array([number_of_channels])

    Test_Results['HFO_found'] = np.zeros(number_of_channels)
    Test_Results['HFO_rates'] = np.zeros(number_of_channels)
    Test_Results['HFO_periods'] = {}

    # ========================================================================================
    # Loop over pre recorded channels
    # ========================================================================================
    for current_channel in range(number_of_channels):

        Ecog_signal_raw = Data['%s_Ecog_signal' %
                               recording_prefix][current_channel, :]
        HFO_mark = Data['pattern_%s' % recording_type][current_channel]
        time_vector = Data['time_%s' % recording_type][0]
        label = Data['labels_%s' % recording_type][0][current_channel][0]

        Ecog_signal, Ecog_spikes = pre_process_data(raw_signal=Ecog_signal_raw,
                                                    time_vector=time_vector,
                                                    sampling_frequency=sampling_frequency,
                                                    interpfact=interpfact,
                                                    refractory=refractory,
                                                    FR_scaling_factor=FR_scaling_factor)

        Ecog_signal['teacher'] = HFO_mark
        Ecog_signal['label'] = label

        Test_Results['Info']['Recording_duration'] = np.array(
            [time_vector[-1]])
        Test_Results['Info']['labels'] = Data['labels_%s' % recording_type][0]

        # ===============================================================================================================
        # Configure spikes for network input
        # ===============================================================================================================
        spikes_list = {}
        spikes_list['up'] = Ecog_spikes['up']
        spikes_list['dn'] = Ecog_spikes['dn']
        spiketimes, neurons_id = concatenate_spikes(spikes_list)

        # ===============================================================================================================
        # RUN SNN
        # ===============================================================================================================
        # ==============================================================================
        # Input Neurons
        # ==============================================================================
        start_scope()

        Input = SpikeGeneratorGroup(Input_channels,
                                    neurons_id,
                                    spiketimes*second,
                                    dt=100*us, name='Input')

        # ==============================================================================
        # Hidden layer
        #==============================================================================
        equation_path = os.path.join('SNN_HFO_ECoG', 'Equations')
        builder_object1 = NeuronEquationBuilder.import_eq(os.path.join(equation_path, 'Neuron_model'), num_inputs=1)
        Hidden_layer = Neurons(Hidden_neurons, equation_builder = builder_object1, name = 'Hidden_layer', dt=100*us) 
        Hidden_layer.refP = refractory * second
        Hidden_layer.Itau = 3.5e-12 * amp  # 15.3 ms

        # ==============================================================================
        # Input - Hidden layer Synapses
        #==============================================================================
        builder_object2 = SynapseEquationBuilder.import_eq(os.path.join(equation_path, 'Synapse_model'))
        Input_Hidden_layer = Connections(Input, Hidden_layer, equation_builder = builder_object2, name='Input_Hidden_layer', verbose=False, dt=100*us)

        # Connect
        Input_Hidden_layer.connect()
        Input_Hidden_layer.weight = input_hidden_weights
        Input_Hidden_layer.I_tau = getTauCurrent(
            input_hidden_time_constants*1e-3, True) * amp
        Input_Hidden_layer.baseweight = 1 * pamp

        # ==============================================================================
        # Monitors
        # ==============================================================================
        # Spike monitors
        Spike_Monitor_Hidden = SpikeMonitor(Hidden_layer)

        # Neuron monitors
        Hidden_Monitor = StateMonitor(Hidden_layer, variables=['Iin'],
                                      record=[0], name='Hidden_Monitor')

        # ==============================================================================
        # Running simulation
        # ==============================================================================
        duration = np.max(Ecog_signal['time']) + extra_simulation_time
        #duration = 0.01
        print('Running SNN for Patient %s channel %s ' %
              (current_patient, current_channel))
        print('Signal time is ', duration, ' seconds')
        run(duration * second)

        # ==============================================================================
        # RUN HFO analysis
        # ==============================================================================
        # ==============================================================================
        # Upsampling teacher
        # ==============================================================================
        test_teacher = Ecog_signal['teacher']
        test_time = Ecog_signal['time']
        start_teacher = []
        stop_teacher = []
        new_test_teacher = np.zeros((Hidden_Monitor.t/second).size)

        for m in range(test_teacher.size-1):
            if m == 0 and test_teacher[0] == 1:
                start_teacher.append(m)
            if m > 0 and test_teacher[m] == 1 and test_teacher[m-1] == 0:
                start_teacher.append(m)
            elif test_teacher[m] == 1 and test_teacher[m+1] == 0:
                stop_teacher.append(m)
            if m == test_teacher.size-2 and test_teacher[m] == 1:
                stop_teacher.append(m)

        start_teacher = np.asarray(start_teacher)
        stop_teacher = np.asarray(stop_teacher)

        for m in range(start_teacher.size):
            new_test_teacher[np.where(np.logical_and((Hidden_Monitor.t/second) >= test_time[start_teacher[m]],
                                                     (Hidden_Monitor.t/second) <= test_time[stop_teacher[m]]))] = 1

        # ==============================================================================
        # Count HFO and noise detections
        # ==============================================================================
        step_size = 0.01
        window_size = 0.05

        start_teacher = []
        stop_teacher = []
        neurons_spiking_at_HFO = []
        neurons_spiking_at_noise = []

        for m in range(new_test_teacher.size-1):
            if m > 0 and new_test_teacher[m] == 1 and new_test_teacher[m-1] == 0:
                start_teacher.append(m)
            elif new_test_teacher[m] == 1 and new_test_teacher[m+1] == 0:
                stop_teacher.append(m)

        start_teacher = np.asarray(start_teacher)
        stop_teacher = np.asarray(stop_teacher)

        for m in range(start_teacher.size):
            range_of_spikes_HFO = np.where(np.logical_and((Spike_Monitor_Hidden.t/second) >= (Hidden_Monitor.t/second)[start_teacher[m]],
                                                          (Spike_Monitor_Hidden.t/second) <= (Hidden_Monitor.t/second)[stop_teacher[m]]))
            if m == 0:
                range_of_spikes_noise = np.where(np.logical_and((Spike_Monitor_Hidden.t/second) > (Hidden_Monitor.t/second)[0],
                                                                (Spike_Monitor_Hidden.t/second) < (Hidden_Monitor.t/second)[start_teacher[m]]))
            else:
                range_of_spikes_noise = np.where(np.logical_and((Spike_Monitor_Hidden.t/second) > (Hidden_Monitor.t/second)[stop_teacher[m-1]],
                                                                (Spike_Monitor_Hidden.t/second) < (Hidden_Monitor.t/second)[start_teacher[m]]))

            neurons_spiking_at_HFO.append(
                ((Spike_Monitor_Hidden.i/1)[range_of_spikes_HFO]).size)
            neurons_spiking_at_noise.append(
                ((Spike_Monitor_Hidden.i/1)[range_of_spikes_noise]).size)

        neurons_spiking_at_HFO = np.asarray(neurons_spiking_at_HFO)
        neurons_spiking_at_noise = np.asarray(neurons_spiking_at_noise)
        # ===============================================================================================================
        # HFO detection
        # ===============================================================================================================
        print('Running HFO detection')
        HFO_detection = detect_HFO(trial_duration=duration,
                                   spike_monitor=(
                                       Spike_Monitor_Hidden.t/second),
                                   neuron_spike_monitor=(
                                       Spike_Monitor_Hidden.i/1),
                                   step_size=step_size,
                                   window_size=window_size)

        Test_Results['HFO_found'][current_channel] = HFO_detection['total_HFO']
        Test_Results['HFO_rates'][current_channel] = HFO_detection['total_HFO']/duration

        print(current_channel)
        Test_Results['HFO_periods']['channel_%s' %
                                    current_channel] = HFO_detection['periods_HFO']
        del Spike_Monitor_Hidden, Hidden_Monitor
        # ===============================================================================================================
        # SNN Test report
        # ===============================================================================================================
        print('True_positives', np.nonzero(neurons_spiking_at_HFO)[0].size)
        print('False_positives', np.nonzero(neurons_spiking_at_noise)[0].size)
        print('True HFO', start_teacher.size)
        print('Found HFO', HFO_detection['total_HFO'])
        print(' ')

    #sio.savemat(save_path_results + 'Test_Results_SNN_P%s.mat' %current_patient, Test_Results, oned_as = 'row')
    #del Test_Results
