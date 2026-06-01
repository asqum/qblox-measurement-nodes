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
from single_qubit_experiment_helpers.experiment import Experiment
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
from ramsey_vs_flux import MultiplexedRamseyVsFlux
from allxy import MultiplexedAllXY
from drag_pulse_calibration import MultiplexedDRAGCalibration
from defect_spectroscopy import MultiplexedDefectSpectroscopy

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
    hardware_configuration="./dependencies/configs/hw_config_fieldlab.json",
    quantum_device_configuration="./dependencies/configs/dut_config_fieldlab.json"
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
cluster['cluster'].reset()

# %% [markdown]
# ## Example of how to add a qubit to the config file

# %%
# 1. Dump q3's parameters into a dictionary
q4_data = q3.model_dump()

# 2. Update the name and hardware port mappings to match the 'q4' convention
q4_data["name"] = "q4"
q4_data["element_type"] = "FluxTunableTransmonElement"

q4_data["ports"]["microwave"] = "q4:mw"
q4_data["ports"]["flux"] = "q4:fl"
q4_data["ports"]["readout"] = "q4:res"

# 3. Inject your custom frequencies (in Hz)
q4_data["clock_freqs"]["f01"] = 6.26e9
q4_data["clock_freqs"]["readout"] = 7.9e9

# 4. Instantiate the new qubit and add it to the quantum device
q4 = FluxTunableTransmonElement(**q4_data)
hw_agent.quantum_device.add_element(q4)

# 5. Append to your working list
qubits.append(q4)

# Verify it worked
print(f"Successfully created {q4.name}!")
print(f" -> f01: {q4.clock_freqs.f01 / 1e9:.3f} GHz")
print(f" -> Readout: {q4.clock_freqs.readout / 1e9:.3f} GHz")
print(f" -> Inherited amp180: {q4.rxy.amp180:.3f} V")

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
dut_file_name = "dut_config_fieldlab"

hw_agent.quantum_device.name = dut_file_name
hw_agent.quantum_device.to_json_file("./dependencies/configs", add_timestamp=False)

native_file = f"./dependencies/configs/{dut_file_name}.json"
viewable_file = f"./dependencies/configs/{dut_file_name}_viewable.json"

create_viewable_json(native_file, viewable_file)

# %%
for q in qubits:
    print(q.measure.integration_time)
    print(q.measure.pulse_duration)

# %%
# Re-fetch the freshly upgraded qubits from the device!
q0 = hw_agent.quantum_device.get_element("q0")
q1 = hw_agent.quantum_device.get_element("q1")
q2 = hw_agent.quantum_device.get_element("q2")
q3 = hw_agent.quantum_device.get_element("q3")
q4 = hw_agent.quantum_device.get_element("q4")

# UPDATE THE LIST to hold the new objects!
qubits = [q0, q1, q2, q3, q4]

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
cluster['cluster'].module5.in0_att()

# %%
cluster['cluster'].module5.in0_att(0)
cluster['cluster'].module5.out0_att(30)

# %%
cluster['cluster'].module5.in0_att()


# %%
set_attenuation(hw_agent, qubits, 'ro', 2)

# Initialize targeting Module x for the NCO update
tof = TimeOfFlight([q0], module_number=5)

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
        playback_delay = 146e-9)
tof.plot_analysis()

# Applies the snapped delays back to the hardware agent
tof.post_run(qubits_to_update=qubits)

# %%
for q in qubits[0:4]:
    q.measure.acq_delay = qubits[0].measure.acq_delay
    print(f'acquisition delay {q.name} = {q.measure.acq_delay/1e-9} ns')


# %%
# Plot Time of Flight Pulse Diagram
tof_dummy = TimeOfFlight([q0], module_number=5)
tof_dummy.execute(
    frequency_detuning=50e6, 
    pulse_duration=1e-6, 
    pulse_amp=0.7, 
    acquisition_duration=2e-6, 
    repetitions=1
)
compiled_schedule = tof_dummy.compile()
compiled_schedule.plot_pulse_diagram(plot_backend='plotly')

# %%
compiled_schedule.compiled_instructions

# %% [markdown]
# # Resonator spectroscopy measurements

# %%
# -------------------------------------------------------------------------
# TWPA Pump Configuration
# -------------------------------------------------------------------------
# This block configures module 11 to output a continuous wave (CW) pump tone 
# to activate the Traveling Wave Parametric Amplifier (TWPA).

# Connect the sequencer to output 1 using the IQ path
cluster['cluster'].module11.disconnect_outputs()
cluster['cluster'].module11.sequencer5.connect_out1('IQ')

# Enable manual control over the marker (used to toggle a physical switch for the pump line)
cluster['cluster'].module11.sequencer5.marker_ovr_en(False)
cluster['cluster'].module11.sequencer5.marker_ovr_value(3)

# Set the NCO frequency to 0 Hz (we only rely on the LO for the frequency)
cluster['cluster'].module11.sequencer5.nco_freq(-100e6)

# Set the Local Oscillator (LO) to the desired TWPA pump frequency (8.5 GHz)
cluster['cluster'].module11.out1_lo_freq(8.2e9)
# cluster['cluster'].module11.out1_att(10)

# Apply a DC offset to the I-path (path 0) of the mixer. 
# Because the NCO is at 0 Hz, this constant DC offset mixes with the LO 
# to generate a continuous microwave tone at exactly 8.5 GHz. 
# The value (0.2) dictates the amplitude/power of the pump tone.
cluster['cluster'].module11.out1_offset_path0(0)

# %%
# 1. Initialize
rs_wideband = MultiplexedResonatorSpectroscopyFullBandwidth(qubits)

# 2. Execute by simply defining the LO and the resolution
rs_wideband.execute(
    lo_freq=7.6e9,          # The class maps 1 GHz around this LO
    frequency_npoints=500,  # Points per qubit's slice
    repetitions=200,
    ro_amp=0.3,             
    ro_att=30               
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
set_attenuation(hw_agent, qubits, 'ro', 30)

# %%
rs_multi = MultiplexedResonatorSpectroscopy(qubits)

rs_multi.execute(
    frequency_width=4e6, 
    frequency_npoints=500, 
    repetitions=200,
    ro_amp=0.2,
    ro_att=30 
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
    amp_stop=0.24, 
    amp_nsteps=11, 
    repetitions=100,
    ro_att= 30
)

rs_punch_amp.analyze()
rs_punch_amp.plot_analysis()


# %%
# Overwrite hardware config based on visuals
rs_punch_amp.post_run({"q0": 0.1, "q1": 0.15, "q2": 0.40})

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
    flux_start=-0.3,          
    flux_stop=0.3,            
    flux_step=0.03,           
    repetitions=100 
)

flux_spec_multi.analyze()
flux_spec_multi.plot_analysis()

# %%
flux_spec_multi.post_run()

# %%
# read and update qubit parameters from device config
q0 = hw_agent.quantum_device.get_element("q0")
q1 = hw_agent.quantum_device.get_element("q1")
q2 = hw_agent.quantum_device.get_element("q2")
q3 = hw_agent.quantum_device.get_element("q3")
qubits = [q0,q1,q2,q3]

# %%
for q in qubits:
    print(f'Flux sweet spot {q.name}: {q.flux_params.sweet_spot} V')

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
q2.clock_freqs.f01 = 5.279831e9

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
    "q0": 18,
    "q1": 18,
    "q2": 18,
    "q3": 0,

}

target_duration={
        "q0": 100e-9, 
        "q1": 100e-9,  
        "q2": 100e-9,  
        "q3": 200e-9,  
    }

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
qubits = [q3]

# %%
pulsed_flux_spec = MultiplexedPulsedFluxQubitSpectroscopy(qubits)

# Sweep fluxaround the sweet spot, and sweep freq
pulsed_flux_spec.execute(
    flux_span=0.06,
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
    frequency_detuning=2e6,
    repetitions=500
)

ramsey_multi.analyze()
# ramsey_multi.plot_iq()
ramsey_multi.plot_analysis()



# %%
# ramsey_multi.post_run(sign_overrides={"q3": -1}, qubits_to_update=["q3"])
ramsey_multi.post_run()

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
    n_values=[1,3,5, 7, 9, 11, 13], 
    repetitions=200
)

# 3. Fit the 2D surface to calculate the amplitude scale factors
pi_pulse_error_amplification.analyze()

# 4. Plot the heatmap and the classic "bowtie" intersections
pi_pulse_error_amplification.plot_analysis()



# %%
# 5. Update the device configuration
# You can leave it empty to update all successful fits: pi_pulse_error_amplification.post_run()
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
    amp_rel_span=0.1, 
    amp_npoints=31, 
    n_values=[1, 2, 3, 4, 5, 6, 7], 
    repetitions=300
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

# %%
# read and update qubit parameters from device config
q0 = hw_agent.quantum_device.get_element("q0")
q1 = hw_agent.quantum_device.get_element("q1")
q2 = hw_agent.quantum_device.get_element("q2")
q3 = hw_agent.quantum_device.get_element("q3")
qubits = [q3]

# %% [markdown]
# # T2 as a function of flux

# %%
# Initialize the experiment
ramsey_flux = MultiplexedRamseyVsFlux([q3])

# Execute the 2D Sweep
ramsey_flux.execute(
    tau_start=1e-6, 
    tau_stop=20e-6,
    tau_step=180e-9,  
    frequency_detuning=0.1e6,   #  artificial detuning
    flux_span=0.015,           # Sweeps +/- around current sweet spot
    flux_npoints=10,          # flux slices
    repetitions=200           # Shots per point
)

# Fit the S21 traces to find T2*, then fit the T2* array to find the peak
ramsey_flux.analyze()

# Plot the 2D Heatmap and the 1D Parabola
ramsey_flux.plot_analysis()

# Automatically update the qubits with their new maximized sweet spot
# ramsey_flux.post_run()

# %% [markdown]
# # Echo

# %%
echo_multi = MultiplexedEcho(qubits)
echo_multi.execute(
    tau_start=1e-6, 
    tau_stop=100e-6, 
    tau_step=2e-6, 
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
    q.measure.pulse_duration = 20e-6
    q.measure.integration_time = 20e-6

# %%
qubits = [q1]

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

# Sweep around center frequency
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
cluster['cluster'].module8.sequencer5.marker_ovr_value(3)
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
# # AllXY

# %%
allxy_exp = MultiplexedAllXY(qubits)

# Execute
allxy_exp.execute(repetitions=500)

allxy_exp.analyze()
allxy_exp.plot_analysis()


# %% [markdown]
# # DRAG Calibration

# %%
def generate_drag(t, amplitude, beta, duration, nr_sigma=4):
    """Calculates the I and Q components of a DRAG pulse."""
    mu = t[0] + duration / 2
    sigma = duration / (2 * nr_sigma)
    
    # I-component (Gaussian)
    gauss_env = amplitude * np.exp(-(0.5 * ((t - mu) ** 2) / sigma**2))
    
    # Q-component (Derivative)
    deriv_gauss_env = -beta * (t - mu) / sigma**2 * gauss_env
    
    return gauss_env, deriv_gauss_env

# 1. Define pulse parameters
duration = q3.rxy.duration  # 40 ns pulse
t = np.linspace(0, duration, 1000)
amplitude = q3.rxy.amp180

# 2. Generate waveforms for positive and negative beta
I_base, Q_pos = generate_drag(t, amplitude, beta=1e-8, duration=duration)
_     , Q_neg = generate_drag(t, amplitude, beta=-1e-8, duration=duration)

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Plot Beta = +1e-8
ax1.plot(t * 1e9, I_base, 'k--', lw=2, label='I (Real) - Gaussian')
ax1.plot(t * 1e9, Q_pos, 'b-', lw=2, label=r'Q (Imag) - DRAG ($\beta = +10^{-8}$)')
ax1.set_title(r"DRAG Pulse ($\beta > 0$)")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Amplitude (a.u.)")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower center')

# Plot Beta = -1e-8
ax2.plot(t * 1e9, I_base, 'k--', lw=2, label='I (Real) - Gaussian')
ax2.plot(t * 1e9, Q_neg, 'r-', lw=2, label=r'Q (Imag) - DRAG ($\beta = -10^{-8}$)')
ax2.set_title(r"DRAG Pulse ($\beta < 0$)")
ax2.set_xlabel("Time (ns)")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower center')

plt.tight_layout()
plt.show()

# %%
# 1. Initialize the experiment
drag_cal = MultiplexedDRAGCalibration(qubits)

# 2. Execute the sweep
drag_cal.execute(
    beta_start=-8e-9, 
    beta_stop=5e-9, 
    beta_npoints=41, 
    pulse_repetitions=1,  # Try 3 or 5 if the lines are too flat!
    repetitions=500
)

# 3. Analyze and plot the "bowtie"
drag_cal.analyze()
drag_cal.plot_analysis()



# %%
# 4. Save the optimal beta back to the device config
drag_cal.post_run()

# %% [markdown]
# # Defect state spectroscopy

# %%
# 1. Initialize the experiment with your target qubits
defect_spec = MultiplexedDefectSpectroscopy(qubits)

# 2. Execute the 2D Sweep
# We sweep a wide flux range to scan the qubit's entire frequency band, 
# and sweep the interaction time (tau) up to 1 microsecond to capture 
# multiple vacuum Rabi oscillations.
defect_spec.execute(
    flux_start=-0.3, 
    flux_stop=0.3, 
    flux_npoints=101,     # High resolution to catch narrow defects
    tau=100e-9,           # The fixed 100ns wait time from the paper
    repetitions=500      # High averages to drop the noise floor
)

# 3. Analyze the S21 data (automatically applies PCA to project the SWAP signal)
defect_spec.analyze()

# 4. Plot 
defect_spec.plot_analysis()

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
# 1 seed and one length can give 3000 length sequence
lengths = np.arange(0, 200, 40)
seeds = np.random.randint(0, 2**31 - 1, size=10, dtype=np.int32)

repetitions = 1

# %%
qubit = q3

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
