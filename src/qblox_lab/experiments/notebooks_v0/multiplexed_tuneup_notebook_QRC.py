# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
# Enable autoreload to automatically update changes made in .py files
# %load_ext autoreload
# %autoreload 2

# # # ACTIVATE INTERACTIVE MATPLOTLIB
# # %matplotlib widget


import time
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


# Helpers and Setup
from helpers import create_viewable_json
from analysis import set_attenuation
from riken_helpers.experiment import Experiment
from qblox_scheduler import HardwareAgent
from schedules import print_instructions
from custom_elements import FluxTunableTransmonElement

# Experiment Classes
from time_of_flight import TimeOfFlight
from resonator_spectroscopy_full_bandwidth import MultiplexedResonatorSpectroscopyFullBandwidth
from resonator_spectroscopy import MultiplexedResonatorSpectroscopy
from resonator_punchout import MultiplexedResonatorPunchout, MultiplexedResonatorPunchoutAmp
from resonator_flux_spectroscopy import MultiplexedResonatorFluxSpectroscopy
from qubit_spectroscopy import MultiplexedQubitSpectroscopy
from power_rabi import MultiplexedPowerRabi
from pulsed_flux_qubit_spectroscopy import MultiplexedPulsedFluxQubitSpectroscopy
from t1 import MultiplexedT1
from ramsey import MultiplexedRamsey
from echo import MultiplexedEcho
from dispersive_shift import MultiplexedDispersiveShift
from ssro import MultiplexedSSRO, MultiplexedReadoutAmplitudeOptimization,MultiplexedReadoutFrequencyOptimization
from active_reset import MultiplexedActiveReset
from readout_amplitude_calibration import MultiplexedReadoutAmplitudeCalibration
from pi_pulse_error_amplification import MultiplexedPiPulseErrorAmplification
from pi_half_pulse_error_amplification import MultiplexedPiHalfPulseErrorAmplification

# %%
import matplotlib.pyplot as plt

# =============================================================================
# GLOBAL PLOT SETTINGS
# =============================================================================
# Change the font family (e.g., 'sans-serif', 'serif', 'monospace')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans'] 

# Change font sizes globally
plt.rcParams['font.size'] = 12           # Default text size
plt.rcParams['axes.labelsize'] = 14      # X and Y label size
plt.rcParams['axes.titlesize'] = 14      # Title size
plt.rcParams['xtick.labelsize'] = 11     # X tick label size
plt.rcParams['ytick.labelsize'] = 11     # Y tick label size

# Change axis and tick thickness
plt.rcParams['axes.linewidth'] = 2.0      # Thickness of the plot box
plt.rcParams['xtick.major.width'] = 2.0   # Thickness of X ticks
plt.rcParams['ytick.major.width'] = 2.0   # Thickness of Y ticks
plt.rcParams['xtick.minor.width'] = 1.5   # Thickness of minor X ticks
plt.rcParams['ytick.minor.width'] = 1.5   # Thickness of minor Y ticks

# (Optional) Make the ticks point inwards instead of outwards
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# %% [markdown]
# ## Connect to cluster and create hardware agent

# %%
hw_agent = HardwareAgent(
    hardware_configuration="./dependencies/configs/hw_config_fieldlab_QRC.json",
    quantum_device_configuration="./dependencies/configs/dut_flux_tunable_QRC.json"
)

Experiment.hw_agent = hw_agent
Experiment.quantum_device = hw_agent.quantum_device

q0 = hw_agent.quantum_device.get_element("q0")
q1 = hw_agent.quantum_device.get_element("q1")
q2 = hw_agent.quantum_device.get_element("q2")
q3 = hw_agent.quantum_device.get_element("q3")
qubits = [q0,q1,q2,q3]

cluster = hw_agent.get_clusters()
hw_options = hw_agent.hardware_configuration.hardware_options

# %%
cluster['cluster'].get_connected_modules()

# %%
cluster['cluster'].reset()

# %%
cluster['cluster'].module8.__dict__

# %% [markdown]
# # Save configs

# %%
# Save it directly. Let it keep the NaNs!
dut_file_name = "dut_flux_tunable_QRC"

hw_agent.quantum_device.name = dut_file_name
hw_agent.quantum_device.to_json_file("./dependencies/configs", add_timestamp=False)

native_file = f"./dependencies/configs/{dut_file_name}.json"
viewable_file = f"./dependencies/configs/{dut_file_name}_viewable.json"

create_viewable_json(native_file, viewable_file)

# %%
# Re-fetch the freshly upgraded qubits from the device!
q0 = hw_agent.quantum_device.get_element("q0")
q1 = hw_agent.quantum_device.get_element("q1")
q2 = hw_agent.quantum_device.get_element("q2")
q3 = hw_agent.quantum_device.get_element("q3")

# UPDATE THE LIST to hold the new objects!
qubits = [q0, q1, q2, q3]

for q in qubits:
    print(f"[{q.name}] Element Type: {q.element_type}")
    print(f"  -> Sweet Spot: {q.flux_params.sweet_spot:.3f} V")
    
    # Safely fetch the values
    amp90_val = getattr(q.pi_half, 'amp90', float('nan'))
    amp180_val = q.rxy.amp180
    naive_half_pi = amp180_val / 2.0
    
    print(f"  -> Pi amp (amp180):         {amp180_val:.6f} V")
    print(f"  -> Naive Pi/2 (amp180 / 2): {naive_half_pi:.6f} V")
    print(f"  -> Calibrated Pi/2 (amp90): {amp90_val:.6f} V")
    
    # Calculate and print the difference if amp90 has been calibrated
    if not math.isnan(amp90_val):
        delta = amp90_val - naive_half_pi
        perc_diff = (delta / naive_half_pi) * 100
        print(f"  -> Calibration Delta:       {delta:+.6f} V ({perc_diff:+.2f}%)")
        
    print("-" * 40)

# %% [markdown]
# # Time of flight
# Prior to the start of quantum experiments, it is crucial to calibrate the acquisition delay.

# %%
hw_options.output_att['q0:res-q0.ro']

# %%
cluster['cluster'].module13.in0_att(20)
cluster['cluster'].module13.out0_att(20)

# %%
cluster['cluster'].module13.out0_att()


# %%
set_attenuation(hw_agent, qubits, 'ro', 0)

# Initialize targeting Module 13 for the NCO update
tof = TimeOfFlight([q0], module_number=13)

# Run the 2 microsecond trace
tof.execute(
    frequency_detuning=50e6, 
    pulse_duration=1e-6, 
    pulse_amp=0.7, 
    acquisition_duration=2e-6, 
    repetitions=100
)

# Math & Visualization
tof.analyze(
        acquisition_delay= 0, 
        playback_delay = 267e-9)
tof.plot_analysis()

# Applies the snapped delays back to the hardware agent
tof.post_run()

# %%
# Plot Time of Flight Pulse Diagram
tof_dummy = TimeOfFlight([q0], module_number=10)
tof_dummy.execute(
    frequency_detuning=50e6, 
    pulse_duration=1e-6, 
    pulse_amp=0.7, 
    acquisition_duration=2e-6, 
    repetitions=1
)
compiled_schedule = tof_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Resonator spectroscopy measurements

# %%
# -------------------------------------------------------------------------
# TWPA Pump Configuration
# -------------------------------------------------------------------------
# This block configures module 8 to output a continuous wave (CW) pump tone 
# to activate the Traveling Wave Parametric Amplifier (TWPA).

# Connect the sequencer to output 1 using the IQ path
cluster['cluster'].module8.disconnect_outputs()
cluster['cluster'].module8.sequencer5.connect_out1('IQ')

# Enable manual control over the marker (used to toggle a physical switch for the pump line)
cluster['cluster'].module8.sequencer5.marker_ovr_en(True)
cluster['cluster'].module8.sequencer5.marker_ovr_value(2)

# Set the NCO frequency to 0 Hz (we only rely on the LO for the frequency)
cluster['cluster'].module8.sequencer5.nco_freq(0)

# Set the Local Oscillator (LO) to the desired TWPA pump frequency (8.5 GHz)
cluster['cluster'].module8.out1_lo_freq(8.5e9)
cluster['cluster'].module8.out0_lo_freq(8.5e9)

# Apply a DC offset to the I-path (path 0) of the mixer. 
# Because the NCO is at 0 Hz, this constant DC offset mixes with the LO 
# to generate a continuous microwave tone at exactly 8.5 GHz. 
# The value (0.2) dictates the amplitude/power of the pump tone.
cluster['cluster'].module8.out1_offset_path0(0.2)

# %%
# 1. Initialize
rs_wideband = MultiplexedResonatorSpectroscopyFullBandwidth(qubits)

# 2. Execute by simply defining the LO and the resolution
rs_wideband.execute(
    lo_freq=7.8e9,          # The class maps 1 GHz around this LO
    frequency_npoints=1000,  # Points per qubit's 250 MHz slice
    repetitions=200,
    ro_amp=0.2,             
    ro_att=20               
)

# 3. Analyze and Plot
rs_wideband.analyze()
rs_wideband.plot_1d()

# %%
# Plot Wideband Resonator Spectroscopy Pulse Diagram
rs_wideband_dummy = MultiplexedResonatorSpectroscopyFullBandwidth(qubits)
rs_wideband_dummy.execute(lo_freq=7.5e9, frequency_npoints=5, repetitions=1, ro_amp=0.3, ro_att=30)
compiled_schedule = rs_wideband_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %%
rs_multi = MultiplexedResonatorSpectroscopy(qubits)

rs_multi.execute(
    frequency_width=10e6, 
    frequency_npoints=500, 
    repetitions=200,
    ro_amp=0.2,
    ro_att=20 
)

rs_multi.analyze()
rs_multi.plot_analysis()

# %%
for q in qubits:
    print(f'{q.name} readout amplitude = {q.measure.pulse_amp}')

# %%
rs_multi.post_run()

# %%
rs_multi.dataset

# %%
# Plot Narrowband Resonator Spectroscopy Pulse Diagram
rs_multi_dummy = MultiplexedResonatorSpectroscopy(qubits)
rs_multi_dummy.execute(frequency_width=10e6, frequency_npoints=5, repetitions=1)
compiled_schedule = rs_multi_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Resonator punchout

# %%
rs_punch = MultiplexedResonatorPunchout(qubits)
rs_punch.execute(
    frequency_width=10e6, 
    frequency_npoints=200, 
    att_start=10, 
    att_stop=30, 
    att_step=2, 
    repetitions=100,
    ro_amp=0.3
)

rs_punch.analyze()
rs_punch.plot_analysis()

# Overwrite hardware config based on visuals
# rs_punch.post_run(readout_attenuation=16)

# %%
# Plot Resonator Punchout (Att) Pulse Diagram
rs_punch_dummy = MultiplexedResonatorPunchout(qubits)
rs_punch_dummy.execute(frequency_width=10e6, frequency_npoints=2, att_start=10, att_stop=12, att_step=2, repetitions=1)
compiled_schedule = rs_punch_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %%
cluster['cluster'].module10.out0_att()

# %%
rs_punch_amp = MultiplexedResonatorPunchoutAmp(qubits)
rs_punch_amp.execute(
    frequency_width=10e6, 
    frequency_npoints=200, 
    amp_start=0.02, 
    amp_stop=0.9, 
    amp_nsteps=15, 
    repetitions=100,
    ro_att= 30
)

rs_punch_amp.analyze()
rs_punch_amp.plot_analysis()

# Overwrite hardware config based on visuals
# rs_punch_amp.post_run(readout_amplitude=0.3)

# %%
# Plot Resonator Punchout (Amp) Pulse Diagram
rs_punch_amp_dummy = MultiplexedResonatorPunchoutAmp(qubits)
rs_punch_amp_dummy.execute(frequency_width=10e6, frequency_npoints=2, amp_start=0.1, amp_stop=0.2, amp_nsteps=2, repetitions=1)
compiled_schedule = rs_punch_amp_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Resonator flux spectroscopy

# %%
flux_spec_multi = MultiplexedResonatorFluxSpectroscopy(qubits)

flux_spec_multi.execute(
    frequency_width=5e6,       
    frequency_npoints=150,      
    flux_start=-0.25,          
    flux_stop=0.25,            
    flux_step=0.03,           
    repetitions=100 
)

flux_spec_multi.analyze()
flux_spec_multi.plot_analysis()

# %%
flux_spec_multi.post_run()

# %%
# Plot Flux Spectroscopy Pulse Diagram
flux_spec_dummy = MultiplexedResonatorFluxSpectroscopy(qubits)
flux_spec_dummy.execute(frequency_width=5e6, frequency_npoints=2, flux_start=-0.5, flux_stop=0.5, flux_step=0.1, repetitions=1)
compiled_schedule = flux_spec_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Qubit spectroscopy

# %%
# 1. Initialize the experiment with your hardware qubits
qubit_spec_multi = MultiplexedQubitSpectroscopy(qubits)

# 2. Execute the schedule on the hardware
qubit_spec_multi.execute(
    f01_width=200e6, 
    f01_npoints=500, 
    voltage_offset=0.03, 
    repetitions=500,
    drive_att=20
)

# 3. Process the data (PCA rotation and Lorentzian fit)
qubit_spec_multi.analyze()

# 4. Generate the plots directly from the class
qubit_spec_multi.plot_analysis() 
# qubit_spec_multi.plot_iq()       

# %%
qubit_spec_multi.post_run()

# %%
# Plot Qubit Spectroscopy Pulse Diagram
qubit_spec_dummy = MultiplexedQubitSpectroscopy(qubits)
qubit_spec_dummy.execute(f01_width=200e6, f01_npoints=5, voltage_offset=0.03, repetitions=1)
compiled_schedule = qubit_spec_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Rabi

# %%
# Define dictionaries for specific qubits
target_att = {
    "q0": 6,
    "q2": 8,
    "q3": 6,
    "q1": 6
}

target_duration = 50e-9 

power_rabi_multi = MultiplexedPowerRabi(qubits)
power_rabi_multi.execute(
    amp_start=0, 
    amp_stop=0.9, 
    amp_npoints=50, 
    repetitions=500,
    drive_att=target_att,         
    drive_duration=target_duration 
)

power_rabi_multi.analyze()
power_rabi_multi.plot_analysis()
# power_rabi_multi.plot_iq()



# %%
power_rabi_multi.post_run()

# %%
# Plot Power Rabi Pulse Diagram
power_rabi_dummy = MultiplexedPowerRabi(qubits)
power_rabi_dummy.execute(amp_start=0, amp_stop=0.5, amp_npoints=2, repetitions=1)
compiled_schedule = power_rabi_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Flux Qubit Spectroscopy

# %%
pulsed_flux_spec = MultiplexedPulsedFluxQubitSpectroscopy(qubits)

# Sweep fluxaround the sweet spot, and sweep freq
pulsed_flux_spec.execute(
    flux_start=-0.03, 
    flux_stop=0.03, 
    flux_npoints=10, 
    freq_shift_start=-80e6, 
    freq_shift_stop=80e6, 
    freq_npoints=50, 
    repetitions=100
)

pulsed_flux_spec.analyze()
pulsed_flux_spec.plot_analysis()



# %%
pulsed_flux_spec.post_run()

# %% [markdown]
# # $T_1$

# %%
t1_multi = MultiplexedT1(qubits)
t1_multi.execute(
    tau_start=1e-6, 
    tau_stop=150e-6, 
    tau_step=2e-6, 
    repetitions=500,
)

t1_multi.analyze()
# t1_multi.plot_iq()
t1_multi.plot_analysis()

# %%
# Plot T1 Pulse Diagram
t1_dummy = MultiplexedT1(qubits)
t1_dummy.execute(tau_start=1e-6, tau_stop=3e-6, tau_step=2e-6, repetitions=1)
compiled_schedule = t1_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Ramsey

# %%
ramsey_multi = MultiplexedRamsey(qubits)
ramsey_multi.execute(
    tau_start=1e-6, 
    tau_stop=1.7e-6,
    tau_step=20e-9, 
    frequency_detuning=0,
    repetitions=500
)

ramsey_multi.analyze()
# ramsey_multi.plot_iq()
ramsey_multi.plot_analysis()



# %%
ramsey_multi.post_run(sign_overrides={"q0": -1}, qubits_to_update=["q0"])
# ramsey_multi.post_run()

# %%
ramsey_multi = MultiplexedRamsey(qubits)
ramsey_multi.execute(
    tau_start=1e-6, 
    tau_stop=50e-6,
    tau_step=300e-9, 
    frequency_detuning=0.5e6,
    repetitions=500
)

ramsey_multi.analyze()
# ramsey_multi.plot_iq()
ramsey_multi.plot_analysis()

# ramsey_multi.post_run()

# %%
# Plot Ramsey Pulse Diagram
ramsey_dummy = MultiplexedRamsey(qubits)
ramsey_dummy.execute(tau_start=1e-6, tau_stop=3e-6, tau_step=2e-6, frequency_detuning=5e6, repetitions=1)
compiled_schedule = ramsey_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Error amplification on pi pulse

# %%
cluster['cluster'].module13.out1_att()

# %%
# 1. Initialize the experiment with your list of active qubits
# (Assuming 'qubits' is already defined in your notebook, e.g., qubits = [q0, q1, q2, q3])
pi_pulse_error_amplification = MultiplexedPiPulseErrorAmplification(qubits)

# 2. Execute the sequence
# - amp_rel_span = 0.1: We sweep +/- 10% around the currently calibrated amp180
# - n_values: We strictly use ODD numbers so the pulse sequence always 
#             aims to leave the qubit on the equator of the Bloch sphere
#             (where measurement sensitivity to angle errors is perfectly maximized).
pi_pulse_error_amplification.execute(
    amp_rel_span=0.05, 
    amp_npoints=31, 
    n_values=[1,3,5, 7, 9, 11, 13, 15], 
    repetitions=200
)

# 3. Fit the 2D surface to calculate the amplitude scale factors
pi_pulse_error_amplification.analyze()

# 4. Plot the heatmap and the classic "bowtie" intersections
pi_pulse_error_amplification.plot_analysis()



# %%
# 5. Safely update the device configuration!
# You can leave it empty to update all successful fits: pingpi_pulse_error_amplification_pong.post_run()
# Or you can selectively update just the ones that look clean in the plots:
pi_pulse_error_amplification.post_run()

# %% [markdown]
# # Error amplification on pi/2 pulse

# %%
# Import the class


# 1. Initialize the experiment with your active qubits
# (Assuming 'qubits' is already defined, e.g., qubits = [q0, q1, q2, q3])
pi_half_pulse_error_amplification = MultiplexedPiHalfPulseErrorAmplification(qubits)

# 2. Execute the sequence
# - amp_rel_span = 0.1: Sweeps +/- 10% around the currently calibrated amp180
# - n_values: Number of pi/2 pulse PAIRS. [1, 2, 3, 4, 5, 6, 7] maps to [2, 4, 6, 8, 10, 12, 14] physical pulses.
pi_half_pulse_error_amplification.execute(
    amp_rel_span=0.05, 
    amp_npoints=31, 
    n_values=[1, 2, 3, 4, 5, 6, 7], 
    repetitions=200
)

# 3. Fit the 2D surface to calculate the amplitude scale factors
pi_half_pulse_error_amplification.analyze()

# 4. Plot the heatmap and the "bowtie" intersections
pi_half_pulse_error_amplification.plot_analysis()



# %%
# 5. Safely update the device configuration!
# Because X90 dynamically divides amp180 by 2 in Qblox, this seamlessly updates 
# your global amp180 baseline to make your pi/2 pulses mathematically perfect.

pi_half_pulse_error_amplification.post_run()

# %% [markdown]
# # Echo

# %%
echo_multi = MultiplexedEcho([q1])
echo_multi.execute(
    tau_start=1e-6, 
    tau_stop=50e-6, 
    tau_step=1e-6, 
    frequency_detuning=1e6, 
    repetitions=500
)

echo_multi.analyze()
echo_multi.plot_iq()
echo_multi.plot_analysis()

# %%
# Plot Echo Pulse Diagram
echo_dummy = MultiplexedEcho(qubits)
echo_dummy.execute(tau_start=1e-6, tau_stop=3e-6, tau_step=2e-6, frequency_detuning=0.5e6, repetitions=1)
compiled_schedule = echo_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Dispersive shift

# %%
ds_multi = MultiplexedDispersiveShift(qubits)

ds_multi.execute(
    frequency_width=4e6,  
    frequency_npoints=100, 
    repetitions=400
)

ds_multi.analyze()
ds_multi.plot_analysis()

# %%
ds_multi.post_run()

# %%
# Plot Dispersive Shift Pulse Diagram
ds_dummy = MultiplexedDispersiveShift(qubits)
ds_dummy.execute(frequency_width=5e6, frequency_npoints=2, repetitions=1)
compiled_schedule = ds_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Single shot readout

# %%
for q in qubits:
    print(q.measure.pulse_duration)

# %%
# -------------------------------
# OPTION A: Standard single SSRO
# -------------------------------
ssro = MultiplexedSSRO(qubits)
ssro.execute(repetitions=500)
ssro.analyze()
ssro.plot_analysis()


# %%
ssro.post_run()

# %%
# Plot SSRO Pulse Diagram
ssro_dummy = MultiplexedSSRO(qubits)
ssro_dummy.execute(repetitions=2) # 2 Repetitions to see both |0> and |1> logic
compiled_schedule = ssro_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %%
# -------------------------------
# OPTION B: Optimization Sweep
# -------------------------------
readout_amplitudes = np.linspace(0.1, 0.3, 5)

opt_sweep = MultiplexedReadoutAmplitudeOptimization(qubits)
opt_sweep.execute(readout_amplitudes=readout_amplitudes, repetitions=500)
opt_sweep.plot_analysis()



# %%
# Manually save optimal settings looking at the plot output
opt_sweep.post_run({
    "q0": 0.2,
    "q1": 0.2,
    "q2": 0.2,
    "q3": 0.2
})

# %%
# Initialize
freq_opt = MultiplexedReadoutFrequencyOptimization(qubits)

# Sweep from -3 MHz to +3 MHz in 21 steps (roughly 300 kHz steps)
detunings = np.linspace(-0.4e6, 0.4e6, 9)

# Execute the sweep
freq_opt.execute(freq_detunings=detunings, repetitions=500)

# View the fidelity peaks
freq_opt.plot_analysis()

# # Supply the optimal detunings (in MHz) that you read off the plot axis
# freq_opt.post_run({
#     "q0": -0.3,  # Shifts q0 readout down by 300 kHz
#     "q1": +1.2,  # Shifts q1 readout up by 1.2 MHz
# })

# %% [markdown]
#
# # Active Reset Verification

# %%
active_reset_multi = MultiplexedActiveReset(qubits)

# Execute 1000 repetitions
active_reset_multi.execute(repetitions=1000)

# Extract and Plot the IQ blobs
active_reset_multi.analyze()
active_reset_multi.plot_analysis()



# %%
# Plot Active Reset Pulse Diagram
ar_dummy = MultiplexedActiveReset(qubits)
ar_dummy.execute(repetitions=1)
compiled_schedule = ar_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %% [markdown]
# # Readout Amplitude Calibration (Stark Shift)

# %%
for q in qubits:
    print(f'{q.name} readout amplitude = {q.measure.pulse_amp}')
    print(f'{q.name} readout duration = {q.measure.pulse_duration}')

# %%
ro_amp_cal = MultiplexedReadoutAmplitudeCalibration(qubits)

# Execute the sweep (Takes a moment due to the 2D nature)
ro_amp_cal.execute(
    amp_stop=0.3, 
    amp_npoints=10, 
    freq_shift_start=-80e6, 
    freq_shift_stop=80e6,
    freq_npoints=50,
    repetitions=100
)

# Analyze targeting a given Stark shift
ro_amp_cal.analyze(target_shift_hz=-10e6,fit_amp_limit=0.5)
ro_amp_cal.plot_analysis()



# %%
# Save the new Readout Amplitude!
ro_amp_cal.post_run()

# %% [markdown]
# # TWPA Optimization

# %%
#####
# MULTIPLEXED TWPA Pump Freq. and Power Optimization Sweep
##### 

# -------------------------------------------------------------------------
# 1. Sweep Parameters & Initialization
# -------------------------------------------------------------------------
freq_start = 8.0e9
freq_stop = 8.5e9
freq_points = 4  

power_start = 0.1
power_stop = 0.4
power_points = 4 

ssro_repetitions = 500 

pump_frequencies = np.linspace(freq_start, freq_stop, freq_points)
pump_powers = np.linspace(power_start, power_stop, power_points)

# Initialize dictionaries to hold individual fidelities, and a matrix for the mean
fidelities_dict = {q.name: np.full((power_points, freq_points), np.nan) for q in qubits}
mean_fidelities = np.full((power_points, freq_points), np.nan)

# -------------------------------------------------------------------------
# 2. Initial Hardware Setup
# -------------------------------------------------------------------------
print("Configuring TWPA pump sequencer...")
cluster['cluster'].module8.disconnect_outputs()
cluster['cluster'].module8.sequencer5.connect_out1('IQ')

cluster['cluster'].module8.sequencer5.marker_ovr_en(True)
cluster['cluster'].module8.sequencer5.marker_ovr_value(2)
cluster['cluster'].module8.sequencer5.nco_freq(0)

# -------------------------------------------------------------------------
# 3. Figure Setup for Subplots
# -------------------------------------------------------------------------
num_qubits = len(qubits)
cols = 2
rows = math.ceil(num_qubits / cols)

fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
if num_qubits > 1:
    axes = axes.flatten()
else:
    axes = [axes]

# Hide any unused subplots (e.g., if you have 3 qubits in a 2x2 grid)
for i in range(num_qubits, len(axes)):
    fig.delaxes(axes[i])

fig.suptitle("TWPA Optimization: Individual Readout Fidelities", fontweight='bold', fontsize=16)

# -------------------------------------------------------------------------
# 4. Execution Loop with Live Plotting
# -------------------------------------------------------------------------
total_points = freq_points * power_points
current_point = 1

for j, freq in enumerate(pump_frequencies):
    cluster['cluster'].module8.out1_lo_freq(freq)
    time.sleep(0.005) # LO settle
    
    for i, power in enumerate(pump_powers):
        cluster['cluster'].module8.out1_offset_path0(power)
        time.sleep(0.001) # Offset settle
        
        # Run Multiplexed SSRO
        ssro_exp = MultiplexedSSRO(qubits)
        ssro_exp.execute(repetitions=ssro_repetitions)
        ssro_exp.analyze(silent=True)
        
        # Extract individual fidelities and compute the mean
        current_fids = []
        fid_strings = []
        for q in qubits:
            if q.name in ssro_exp.analyses:
                fid = ssro_exp.analyses[q.name]['fidelity']
            else:
                fid = np.nan 
                
            fidelities_dict[q.name][i, j] = fid
            current_fids.append(fid)
            fid_strings.append(f"{q.name}: {fid*100:.1f}%" if not np.isnan(fid) else f"{q.name}: N/A")
            
        current_mean_fid = np.nanmean(current_fids)
        mean_fidelities[i, j] = current_mean_fid
        
        # --- LIVE PLOT REFRESH ---
        clear_output(wait=True)
        
        for idx, q in enumerate(qubits):
            ax = axes[idx]
            ax.clear()
            
            mesh = ax.pcolormesh(
                pump_frequencies / 1e9, 
                pump_powers, 
                fidelities_dict[q.name] * 100, 
                shading='auto', 
                cmap='RdYlGn',
                vmin=60, 
                vmax=100
            )
            
            ax.set_title(f"{q.name}", fontweight='bold')
            ax.set_xlabel("Pump Frequency (GHz)")
            ax.set_ylabel("Pump Amplitude (V)")
            
            # Only add the colorbar the very first time so it doesn't duplicate
            if current_point == 1:
                fig.colorbar(mesh, ax=ax, label="Fidelity (%)")
        
        fig.tight_layout()
        display(fig)
        
        # Print summary
        print(f"[{current_point}/{total_points}] Freq: {freq/1e9:.3f} GHz | Power: {power:.3f}")
        print(f"  -> Mean: {current_mean_fid*100:.2f}% | " + " | ".join(fid_strings))
        
        current_point += 1

# -------------------------------------------------------------------------
# 5. Cleanup
# -------------------------------------------------------------------------
cluster['cluster'].module8.out1_offset_path0(0.0)
cluster['cluster'].module8.sequencer5.marker_ovr_value(0)
plt.close(fig) 
print("\nSweep complete. TWPA pump turned off.")

# -------------------------------------------------------------------------
# 6. Extract and Report the Optimal Operating Points
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("🏆 OPTIMAL TWPA CONFIGURATION FOUND")
print("=" * 60)

# Global Optimum (Maximizes Mean Fidelity)
best_global_idx = np.unravel_index(np.nanargmax(mean_fidelities), mean_fidelities.shape)
best_freq = pump_frequencies[best_global_idx[1]]
best_power = pump_powers[best_global_idx[0]]

print(f"[GLOBAL OPTIMUM - Maximizing Mean Fidelity]")
print(f"  -> Frequency : {best_freq / 1e9:.4f} GHz")
print(f"  -> Power     : {best_power:.4f} V")
print(f"  -> Mean Fid  : {mean_fidelities[best_global_idx] * 100:.2f}%")
print("-" * 60)

# Individual Best Points
print("Theoretical Best Point for Each Individual Qubit:")
for q in qubits:
    q_fids = fidelities_dict[q.name]
    best_q_idx = np.unravel_index(np.nanargmax(q_fids), q_fids.shape)
    q_best_freq = pump_frequencies[best_q_idx[1]]
    q_best_power = pump_powers[best_q_idx[0]]
    q_best_fid = q_fids[best_q_idx]
    
    # Also grab what this qubit's fidelity was AT the global optimum for comparison
    fid_at_global = q_fids[best_global_idx]
    
    print(f"  [{q.name}] Best: {q_best_freq/1e9:.4f} GHz, {q_best_power:.4f} V (Fid: {q_best_fid*100:.2f}%)")
    print(f"        (At Global Optimum: {fid_at_global*100:.2f}%)")
print("=" * 60)

# %%
# Set the Local Oscillator (LO) to the desired TWPA pump frequency (8.5 GHz)
cluster['cluster'].module8.out1_lo_freq(8.2e9)

# Apply a DC offset to the I-path (path 0) of the mixer. 
# Because the NCO is at 0 Hz, this constant DC offset mixes with the LO 
# to generate a continuous microwave tone at exactly 8.5 GHz. 
# The value (0.2) dictates the amplitude/power of the pump tone.
cluster['cluster'].module8.out1_offset_path0(0.225)

# %% [markdown]
# # Single qubit Randomized Benchmarking

# %%
import numpy as np
from dependencies.analysis_utils import RBAnalysis
from dependencies.randomized_benchmarking.utils import randomized_benchmarking_schedule
from xarray import open_dataset

from qblox_scheduler import HardwareAgent

# %%
np.arange(0, 130, 40)

# %%
# RB settings
lengths = np.arange(0, 130, 40)
seeds = np.random.randint(0, 2**31 - 1, size=10, dtype=np.int32)

repetitions = 1

# %%
qubit = q1

# %%
sched = randomized_benchmarking_schedule(
    qubit.name,
    lengths=lengths,
    repetitions=repetitions,
    seeds=seeds,
)

# Execute the experiment
rb_data = hw_agent.run(sched)

# %%
rb_analysis = RBAnalysis(rb_data).run()
rb_analysis.display_figs_mpl()

# %% [markdown]
# # Cryoscope

# %%
# -------------------------------------------------------------------------
# Cryoscope
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# 1. Imports and Hardware Setup
# -------------------------------------------------------------------------
from cryoscope import CryoscopeExperiment

def main():
    # -------------------------------------------------------------------------
    # 2. Initialize the Experiment
    # -------------------------------------------------------------------------
    print("--- Initializing Cryoscope Measurement ---")
    
    # We pass the list of qubits we want to measure (e.g., just the first one)
    target_qubits = [qubits[0]]
    
    cryo_exp = CryoscopeExperiment(qubits=target_qubits)
    
    # Ensure the hardware agent is attached to the experiment so it can compile and run
    cryo_exp.hw_agent = hw_agent 

    # -------------------------------------------------------------------------
    # 3. Define Sweep Parameters & Execute
    # -------------------------------------------------------------------------
    flux_pulse_amplitude = 0.25  # Volts (The target amplitude of your square pulse)
    duration_start = 0.0         # Seconds
    duration_stop = 150e-9       # Seconds (150 ns is usually plenty to see the tail end)
    duration_points = 151        # Gives exactly 1 ns resolution
    shots = 2000                 # Averaging repetitions
    
    print(f"Sweeping flux pulse duration from {duration_start*1e9} to {duration_stop*1e9} ns...")
    
    cryo_exp.execute(
        amp=flux_pulse_amplitude,
        dur_start=duration_start,
        dur_stop=duration_stop,
        dur_npoints=duration_points,
        repetitions=shots
    )

    # -------------------------------------------------------------------------
    # 4. Analyze & Plot
    # -------------------------------------------------------------------------
    # The analyze method handles the PCA rotation, unwraps the phase, 
    # takes the derivative, and normalizes the step response.
    cryo_exp.analyze()

    # Generate the 3-panel plot (Raw, Phase, Step Response)
    cryo_exp.plot_analysis()

    # -------------------------------------------------------------------------
    # 5. Post-Run / Export
    # -------------------------------------------------------------------------
    # In a full pre-distortion pipeline, you would extract the step_response array 
    # here and feed it into an IIR/FIR filter generator.
    cryo_exp.post_run()
    
    # Example of how you would grab the raw data for custom filter fitting later:
    # step_response_array = cryo_exp.analyses[target_qubits[0].name]['step_response']
    # time_axis = cryo_exp.analyses[target_qubits[0].name]['unique_dur']
    
    print("--- Cryoscope Measurement Complete ---")



# %%
#2D Chevron 

# %% [markdown]
# # 2-Qubit Chevron - Phase Calibration

# %%
# -------------------------------------------------------------------------
# 1. Imports and Hardware Setup
# -------------------------------------------------------------------------

from chevron import ChevronAvoidedCrossing

def main():
    print("--- Initializing 2-Qubit Chevron Measurement ---")
    
    # -------------------------------------------------------------------------
    # 2. Assign Qubit and Coupler Roles
    # -------------------------------------------------------------------------
    # Select the specific qubits and coupler for this CZ gate calibration.
    # Adjust these indices to match the specific pair you are testing on the chip!
    q_control = qubits[0]
    q_target = qubits[1]
    flux_element = couplers[0] # The tunable coupler sitting between q0 and q1
    
    print(f"Control Qubit : {q_control.name}")
    print(f"Target Qubit  : {q_target.name}")
    print(f"Flux Element  : {flux_element.name}")

    # Instantiate the experiment
    chevron_exp = ChevronAvoidedCrossing(
        q_control=q_control, 
        q_target=q_target, 
        flux_element=flux_element, 
        hw_agent=hw_agent
    )

    # -------------------------------------------------------------------------
    # 3. Define Sweep Parameters & Execute
    # -------------------------------------------------------------------------
    # Amplitude sweep (Volts)
    amplitude_start = -0.5
    amplitude_stop = 0.5
    amplitude_points = 41  # 41 points gives nice resolution for the 2D plot

    # Duration sweep (Seconds)
    duration_start = 4e-9  
    duration_stop = 150e-9 
    
    # IMPORTANT: The Qblox sequencer grid operates in multiples of 1ns (or 4ns depending on the operation).
    # Keeping the step at 4ns ensures perfectly clean hardware loops.
    duration_step = 4e-9   

    shots = 1000  # Number of averaging repetitions per pixel on the heatmap

    print("\nStarting 2D Chevron Sweep...")
    print(f"Amplitudes : {amplitude_points} points from {amplitude_start} V to {amplitude_stop} V")
    print(f"Durations  : From {duration_start*1e9} ns to {duration_stop*1e9} ns (Step: {duration_step*1e9} ns)")
    
    # Execute the hardware sweep
    chevron_exp.execute(
        amp_start=amplitude_start,
        amp_stop=amplitude_stop,
        amp_npoints=amplitude_points,
        dur_start=duration_start,
        dur_stop=duration_stop,
        dur_step=duration_step,
        repetitions=shots
    )

    # -------------------------------------------------------------------------
    # 4. Analyze & Plot
    # -------------------------------------------------------------------------
    print("\nAnalyzing dataset...")
    # This locates the maximum variance (the avoided crossing) and fits the swap duration
    chevron_exp.analyze()

    print("Generating plots...")
    # This pops up the 2D Heatmap and the 1D slice of the damped sine wave fit
    chevron_exp.plot_analysis()

    # -------------------------------------------------------------------------
    # 5. Post-Run / Update Hardware Dictionary
    # -------------------------------------------------------------------------
    print("\nUpdating hardware parameters...")
    # This pushes the newly found optimal amplitude and duration back into the hardware agent
    chevron_exp.post_run()
    
    print("--- Chevron Calibration Complete ---")

if __name__ == "__main__":
    main()
