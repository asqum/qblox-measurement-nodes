# ---
# jupyter:
#   files_to_bundle_in_zip_file:
#   - dependencies/datasets/t1.hdf5
#   - dependencies/datasets/time_of_flight.hdf5
#   - dependencies/datasets/ramsey.hdf5
#   - dependencies/datasets/qubit_spectroscopy.hdf5
#   - dependencies/configs/hw_config.json
#   - dependencies/configs/dut_config.json
#   - dependencies/datasets/echo.hdf5
#   - dependencies/analysis_utils.py
#   - dependencies/datasets/single_qubit_randomized_benchmarking.hdf5
#   - dependencies/datasets/rabi.hdf5
#   - dependencies/randomized_benchmarking/clifford_group.py
#   - dependencies/randomized_benchmarking/utils.py
#   - dependencies/datasets/resonator_punchout.hdf5
#   - dependencies/datasets/single_shot_readout.hdf5
#   - dependencies/randomized_benchmarking/randomized_benchmarking.py
#   - dependencies/datasets/resonator_spectroscopy.hdf5
#   - dependencies/randomized_benchmarking/clifford_decompositions.py
#   - dependencies/banner.jpeg
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: files_to_bundle_in_zip_file,is_demo,execute
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=["header_banner"]
# <a href="dependencies/banner.jpeg">
#       <img src="dependencies/banner.jpeg" alt="image0" width="100%">
# </a>

# %% [markdown]
# # Fixed-frequency transmon
# This notebook contains all the required steps for the tuneup of fixed-frequency transmons. We
# show how to connect to the instrument as well as example schedules for tuneup and benchmarking
# of this chip architecture.

# %% execution={"iopub.execute_input": "2026-04-15T12:04:07.376496Z", "iopub.status.busy": "2026-04-15T12:04:07.376030Z", "iopub.status.idle": "2026-04-15T12:04:21.586447Z", "shell.execute_reply": "2026-04-15T12:04:21.585637Z"} tags=["imports", "header_0"]
import matplotlib.pyplot as plt
import numpy as np
from dependencies.analysis_utils import (
    EchoAnalysis,
    PunchoutAnalysis,
    QubitSpectroscopyAnalysis,
    RabiAnalysis,
    RamseyAnalysis,
    RBAnalysis,
    ResonatorSpectroscopyAnalysis,
    SSROAnalysis,
    T1Analysis,
    TimeOfFlightAnalysis,
)
from dependencies.randomized_benchmarking.clifford_group import TwoQubitCliffordCZ
from dependencies.randomized_benchmarking.utils import randomized_benchmarking_schedule
from xarray import open_dataset

from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.experiments import SetHardwareOption
from qblox_scheduler.operations import (
    X90,
    IdlePulse,
    Measure,
    Reset,
    SetClockFrequency,
    VoltageOffset,
    X,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

# %% [markdown] tags=["header_1"]
# ## Setup
# The hardware agent manages the connection to the instrument and ensures that pulses and acquisitions happen over the appropriate input and output channels of the Cluster.
# The cell below creates an instance of the `HardwareAgent` based on the hardware- and device-under-test configuration files in the `./dependencies/configs` folder, allowing us to start doing measurements.
# We also define some convenient aliases to use throughout our measurements.
# For a more thorough discussion of the hardware- and device-under-test configuration files, check out [this tutorial](000_transmon_setup.ipynb).

# %% execution={"iopub.execute_input": "2026-04-15T12:04:21.589547Z", "iopub.status.busy": "2026-04-15T12:04:21.588299Z", "iopub.status.idle": "2026-04-15T12:04:22.265906Z", "shell.execute_reply": "2026-04-15T12:04:22.265068Z"} tags=["header_2"]
# Set up hardware agent, this automatically connects to the instrument
hw_agent = HardwareAgent(
    hardware_configuration="./dependencies/configs/hw_config.json",
    quantum_device_configuration="./dependencies/configs/dut_config.json",
)

# convenience aliases
q0 = hw_agent.quantum_device.get_element("q0")  # Qubits 0 and 2 are measured using QRM-RF + QCM-RF
q2 = hw_agent.quantum_device.get_element("q2")
q3 = hw_agent.quantum_device.get_element("q3")  # Qubit 3 is measured using QRC

cluster = hw_agent.get_clusters()["cluster"]
hw_options = hw_agent.hardware_configuration.hardware_options
qubit = q0

# %% [markdown]
# # Time of flight
# Prior to the start of quantum experiments, it is crucial to calibrate the delay between
# sending a measurement pulse and beginning the acquisition on the instrument. This delay should be
# equal to the time it takes for the readout signal to travel through the fridge and is determined
# by the length of the cables. For the experiment a square pulse is played over the output of the
# readout line, while a trace acquisition is done. The time at which the pulse is detected on the
# acquisition path is determined to be the acquisition delay. This also allows us to set the
# [NCO propagation delay](../../../products/architecture/sequencers/readout.md), which corrects for the
# phase accumulated during the time of flight.

# %% [markdown]
# ## Experiment settings

# %% execution={"iopub.execute_input": "2026-04-15T12:04:22.271410Z", "iopub.status.busy": "2026-04-15T12:04:22.271222Z", "iopub.status.idle": "2026-04-15T12:04:22.274283Z", "shell.execute_reply": "2026-04-15T12:04:22.273617Z"}
# Pulse settings
pulse_attenuation = 0  # dB
pulse_amplitude = 1  # a.u.
pulse_frequency_detuning = 100e6  # Hz
pulse_duration = 300e-9  # ns
acquisition_duration = 1e-6  # Hz

repetitions = 1000

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:04:22.276295Z", "iopub.status.busy": "2026-04-15T12:04:22.276097Z", "iopub.status.idle": "2026-04-15T12:04:22.675158Z", "shell.execute_reply": "2026-04-15T12:04:22.674099Z"}
# Set pulse attenuation
prior_att = hw_options.output_att[f"{qubit.name}:res-{qubit.name}.ro"]
hw_options.output_att[f"{qubit.name}:res-{qubit.name}.ro"] = pulse_attenuation

tof_sched = Schedule("time_of_flight")
tof_sched.add(
    SetClockFrequency(
        clock=qubit.name + ".ro",
        clock_freq_new=qubit.clock_freqs.readout - pulse_frequency_detuning,
    )  # Detune clock 100MHz from expected resonance frequency
)
tof_sched.add(IdlePulse(4e-9))
with tof_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
    tof_sched.add(
        Measure(
            qubit.name,
            acq_protocol="Trace",
            pulse_duration=pulse_duration,
            acq_duration=acquisition_duration,
            acq_channel="S_21",
            acq_delay=0,
        )
    )

# Execute the experiment
tof_data = hw_agent.run(tof_sched)
if cluster.is_dummy:
    example_data = open_dataset("./dependencies/datasets/time_of_flight.hdf5", engine="h5netcdf")
    tof_data = tof_data.update({"S_21": example_data.S_21})

# Reset attenuation
hw_options.output_att[f"{qubit.name}:res-{qubit.name}.ro"] = prior_att

# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:04:22.678104Z", "iopub.status.busy": "2026-04-15T12:04:22.677629Z", "iopub.status.idle": "2026-04-15T12:04:23.482429Z", "shell.execute_reply": "2026-04-15T12:04:23.481798Z"}
tof_analysis = TimeOfFlightAnalysis(tof_data).run()
tof_analysis.display_figs_mpl()
# %% [markdown]
# ## Post-run
# %% execution={"iopub.execute_input": "2026-04-15T12:04:23.489681Z", "iopub.status.busy": "2026-04-15T12:04:23.488722Z", "iopub.status.idle": "2026-04-15T12:04:23.500207Z", "shell.execute_reply": "2026-04-15T12:04:23.499506Z"}
qubit.measure.acq_delay = tof_analysis.quantities_of_interest["tof"]

# Set the NCO propagation delay of the readout line.
cluster.module4.sequencer0.nco_prop_delay_comp_en(True)
cluster.module4.sequencer0.nco_prop_delay_comp(
    tof_analysis.quantities_of_interest["nco_prop_delay"] * 1e9
)

# %% [markdown]
# # Resonator Spectroscopy
# In a resonator spectroscopy experiment, we sweep the frequency of a microwave tone
# applied to the readout line. When the drive tone frequency matches the resonator's
# resonance frequency, a change in the amplitude and/or phase of the reflected
# signal is observed.

# %% [markdown]
# ## Experiment settings
# %% execution={"iopub.execute_input": "2026-04-15T12:04:23.502045Z", "iopub.status.busy": "2026-04-15T12:04:23.501863Z", "iopub.status.idle": "2026-04-15T12:04:23.504746Z", "shell.execute_reply": "2026-04-15T12:04:23.504103Z"}
# Frequency settings
frequency_center = qubit.clock_freqs.readout  # Hz
frequency_width = 5e6  # Hz
frequency_npoints = 300

repetitions = 1000

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:04:23.506458Z", "iopub.status.busy": "2026-04-15T12:04:23.506290Z", "iopub.status.idle": "2026-04-15T12:04:33.009452Z", "shell.execute_reply": "2026-04-15T12:04:33.006263Z"}
spec_sched = Schedule("resonator_spectroscopy")
with (
    spec_sched.loop(arange(0, repetitions, 1, DType.NUMBER)),
    spec_sched.loop(
        linspace(
            start=frequency_center - frequency_width / 2,
            stop=frequency_center + frequency_width / 2,
            num=frequency_npoints,
            dtype=DType.FREQUENCY,
        )
    ) as freq,
):
    spec_sched.add(Measure(qubit.name, freq=freq, coords={"frequency": freq}, acq_channel="S_21"))
    spec_sched.add(IdlePulse(10e-6))  # Let the resonator decay

# Execute the experiment
rs_data = hw_agent.run(spec_sched)
if cluster.is_dummy:
    example_data = open_dataset(
        "./dependencies/datasets/resonator_spectroscopy.hdf5", engine="h5netcdf"
    )
    tof_data = rs_data.update({"S_21": example_data.S_21})

# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:04:33.023193Z", "iopub.status.busy": "2026-04-15T12:04:33.020788Z", "iopub.status.idle": "2026-04-15T12:04:39.966520Z", "shell.execute_reply": "2026-04-15T12:04:39.959931Z"}
resspec_analysis = ResonatorSpectroscopyAnalysis(rs_data).run()
resspec_analysis.display_figs_mpl()

# %% [markdown]
# ## Post-run
# %% execution={"iopub.execute_input": "2026-04-15T12:04:39.979662Z", "iopub.status.busy": "2026-04-15T12:04:39.975574Z", "iopub.status.idle": "2026-04-15T12:04:39.986571Z", "shell.execute_reply": "2026-04-15T12:04:39.984763Z"}
# Update device config
qubit.clock_freqs.readout = resspec_analysis.quantities_of_interest["fr"].nominal_value

# %% [markdown]
# # Resonator punchout
# To verify the presence of a qubit coupled to the resonator, and to optimize the readout
# signal-to-noise ratio (SNR) while preventing back-action on the qubit, we perform a
# "punchout" experiment by sweeping the attenuation of the microwave tone over resonator
# spectroscopy traces. In the high-power regime the resonator
# responds at its bare frequency, while at the low-power regime the resonator frequency is
# dressed by the coupling to the qubit. The crossover between the two is called "resonator
# punchout" (Blais et al., 2021: https://arxiv.org/abs/2005.12667).

# %% [markdown]
# ## Experiment settings
# %% execution={"iopub.execute_input": "2026-04-15T12:04:39.990839Z", "iopub.status.busy": "2026-04-15T12:04:39.989686Z", "iopub.status.idle": "2026-04-15T12:04:40.006841Z", "shell.execute_reply": "2026-04-15T12:04:40.003784Z"}
# Frequency settings
frequency_center = qubit.clock_freqs.readout  # Hz
frequency_width = 5e6  # Hz
frequency_npoints = 300

# Attenuation settings
att_start = 0  # dB
att_stop = 30  # dB
att_step = 2  # dB

repetitions = 100

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:04:40.012196Z", "iopub.status.busy": "2026-04-15T12:04:40.010509Z", "iopub.status.idle": "2026-04-15T12:05:20.718496Z", "shell.execute_reply": "2026-04-15T12:05:20.717036Z"}
po_sched = Schedule("resonator_punchout")

with po_sched.loop(
    arange(start=att_start, stop=att_stop, step=att_step, dtype=DType.NUMBER)
) as amp:
    # Set output attenuation
    po_sched.add(SetHardwareOption("output_att", amp, f"{qubit.name}:res-{qubit.name}.ro"))
    with (
        po_sched.loop(arange(0, repetitions, 1, DType.NUMBER)),
        po_sched.loop(
            linspace(
                start=frequency_center - frequency_width / 2,
                stop=frequency_center + frequency_width / 2,
                num=frequency_npoints,
                dtype=DType.FREQUENCY,
            )
        ) as freq,
    ):
        po_sched.add(
            Measure(
                qubit.name,
                freq=freq,
                coords={"frequency": freq, "amp": amp},
                acq_channel="S_21",
            )
        )
        po_sched.add(IdlePulse(10e-6))  # Let the resonator decay

# Execute the experiment
po_data = hw_agent.run(po_sched)
if cluster.is_dummy:
    example_data = open_dataset(
        "./dependencies/datasets/resonator_punchout.hdf5", engine="h5netcdf"
    )
    po_data = po_data.update({"S_21": example_data.S_21})

# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:05:20.722235Z", "iopub.status.busy": "2026-04-15T12:05:20.722007Z", "iopub.status.idle": "2026-04-15T12:05:24.398633Z", "shell.execute_reply": "2026-04-15T12:05:24.397436Z"}
punchout_analysis = PunchoutAnalysis(po_data).run()
punchout_analysis.display_figs_mpl()

# %% [markdown]
# ## Post-run
# %% execution={"iopub.execute_input": "2026-04-15T12:05:24.401189Z", "iopub.status.busy": "2026-04-15T12:05:24.400995Z", "iopub.status.idle": "2026-04-15T12:05:24.403898Z", "shell.execute_reply": "2026-04-15T12:05:24.403247Z"}
# Update the device config
hw_options.output_att[f"{qubit.name}:res-{qubit.name}.ro"] = 24

# %% [markdown]
# # Qubit spectroscopy
# Two-tone spectroscopy is used to determine the transition frequency between the |0⟩ and
# |1⟩ states. In addition to the tone played on the readout line, a second
# microwave tone is played on the drive line of the qubit. When the qubit drive frequency
# becomes resonant with the qubit's |0⟩ $\rightarrow$ |1⟩ transition, the qubit absorbs energy and
# changes its state. This change is then detected in the amplitude of the
# readout signal at the previously calibrated resonator frequency.

# %% [markdown]
# ## Experiment settings
# %% execution={"iopub.execute_input": "2026-04-15T12:05:24.405471Z", "iopub.status.busy": "2026-04-15T12:05:24.405335Z", "iopub.status.idle": "2026-04-15T12:05:24.407826Z", "shell.execute_reply": "2026-04-15T12:05:24.407253Z"}
# Drive attenuation settings. Should be an even number <= 30
drive_att = 30  # dB

# Drive frequency settings
f01_center = qubit.clock_freqs.f01  # Hz
f01_width = 50e6  # Hz
f01_npoints = 200

repetitions = 1e3

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:05:24.409443Z", "iopub.status.busy": "2026-04-15T12:05:24.409308Z", "iopub.status.idle": "2026-04-15T12:05:27.181389Z", "shell.execute_reply": "2026-04-15T12:05:27.180223Z"}
two_tone_sched = Schedule("two_tone_spectroscopy")
# Set the drive attenuation for the experiment
two_tone_sched.add(SetHardwareOption("output_att", drive_att, f"{qubit.name}:mw-{qubit.name}.01"))

with two_tone_sched.loop(
    linspace(
        start=f01_center - f01_width / 2,
        stop=f01_center + f01_width / 2,
        num=f01_npoints,
        dtype=DType.FREQUENCY,
    )
) as freq:
    # Set a constant tone out of the drive line to probe the f01 frequency
    two_tone_sched.add(VoltageOffset(0.01, 0, port=qubit.ports.microwave, clock=qubit.name + ".01"))
    two_tone_sched.add(SetClockFrequency(clock=qubit.name + ".01", clock_freq_new=freq))

    two_tone_sched.add(Reset(qubit.name))
    with two_tone_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
        two_tone_sched.add(Measure(qubit.name, coords={"frequency": freq}, acq_channel="S_21"))

    # Reset drive line voltage to 0
    two_tone_sched.add(VoltageOffset(0, 0, port=qubit.ports.microwave, clock=qubit.name + ".01"))
    two_tone_sched.add(IdlePulse(4e-9))

# Execute the experiment
qs_data = hw_agent.run(two_tone_sched)
if cluster.is_dummy:
    example_data = open_dataset(
        "./dependencies/datasets/qubit_spectroscopy.hdf5", engine="h5netcdf"
    )
    qs_data = qs_data.update({"S_21": example_data.S_21})

# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:05:27.185954Z", "iopub.status.busy": "2026-04-15T12:05:27.185758Z", "iopub.status.idle": "2026-04-15T12:05:28.124802Z", "shell.execute_reply": "2026-04-15T12:05:28.124013Z"}
qs_analysis = QubitSpectroscopyAnalysis(qs_data).run()
qs_analysis.display_figs_mpl()

# %% [markdown]
# ## Post-run
# %% execution={"iopub.execute_input": "2026-04-15T12:05:28.128231Z", "iopub.status.busy": "2026-04-15T12:05:28.128039Z", "iopub.status.idle": "2026-04-15T12:05:28.130808Z", "shell.execute_reply": "2026-04-15T12:05:28.130230Z"}
# Update device config
qubit.clock_freqs.f01 = qs_analysis.quantities_of_interest["frequency_01"].nominal_value

# %% [markdown]
# # Rabi
# After determining the qubit's |0⟩ $\rightarrow$ |1⟩ transition frequency, a Rabi experiment
# is performed to calibrate the required microwave drive amplitude. The frequency and
# duration of the pulse are kept constant while the amplitude is swept, leading to oscillations
# in the qubit state. The power level that first fully inverts the qubit's population (a
# $\pi$-pulse) is then identified.

# %% [markdown]
# ## Experiment settings
# %% execution={"iopub.execute_input": "2026-04-15T12:05:28.132485Z", "iopub.status.busy": "2026-04-15T12:05:28.132351Z", "iopub.status.idle": "2026-04-15T12:05:28.134893Z", "shell.execute_reply": "2026-04-15T12:05:28.134248Z"}
# Drive attenuation settings. Should be an even number <= 30
drive_att = 12  # dB

# Rabi settings
amp_start = -0.5  # a.u.
amp_stop = 0.5  # a.u.
amp_npoints = 100

repetitions = 1000

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:05:28.136716Z", "iopub.status.busy": "2026-04-15T12:05:28.136454Z", "iopub.status.idle": "2026-04-15T12:05:29.780781Z", "shell.execute_reply": "2026-04-15T12:05:29.779854Z"}
rabi_power_sched = Schedule("power_rabi")
rabi_power_sched.add(SetHardwareOption("output_att", drive_att, f"{qubit.name}:mw-{qubit.name}.01"))

with (
    rabi_power_sched.loop(arange(0, repetitions, 1, DType.NUMBER)),
    rabi_power_sched.loop(
        linspace(start=amp_start, stop=amp_stop, num=amp_npoints, dtype=DType.AMPLITUDE)
    ) as amp,
):
    rabi_power_sched.add(Reset(qubit.name))
    # Play pulse of varying amplitude
    rabi_power_sched.add(X(qubit=qubit.name, amp180=amp))
    rabi_power_sched.add(Measure(qubit.name, coords={"amplitude": amp}, acq_channel="S_21"))

# Execute the experiment
rabi_data = hw_agent.run(rabi_power_sched)
if cluster.is_dummy:
    example_data = open_dataset("./dependencies/datasets/rabi.hdf5", engine="h5netcdf")
    rabi_data = rabi_data.update({"S_21": example_data.S_21})

# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:05:29.783804Z", "iopub.status.busy": "2026-04-15T12:05:29.783657Z", "iopub.status.idle": "2026-04-15T12:05:30.660353Z", "shell.execute_reply": "2026-04-15T12:05:30.658904Z"}
rabi_analysis = RabiAnalysis(rabi_data).run()
rabi_analysis.display_figs_mpl()

# %% [markdown]
# ## Post-run
# %% execution={"iopub.execute_input": "2026-04-15T12:05:30.662271Z", "iopub.status.busy": "2026-04-15T12:05:30.662112Z", "iopub.status.idle": "2026-04-15T12:05:30.664801Z", "shell.execute_reply": "2026-04-15T12:05:30.664146Z"}
# Update device config
qubit.rxy.amp180 = rabi_analysis.quantities_of_interest["Pi-pulse amplitude"].nominal_value

# %% [markdown]
# # $T_1$
# To analyze how quickly a qubit relaxes to the ground state from the excited state, a $T_1$
# experiment is performed. For this type of measurement, the qubit is initialized in the
# |0⟩ state and driven to the |1⟩ state using a $\pi$-pulse. By performing measurements at
# different times $\tau$ after the $\pi$-pulse the decay constant $T_1$ can be measured.

# %% [markdown]
# ## Experiment settings
# %% execution={"iopub.execute_input": "2026-04-15T12:05:30.666353Z", "iopub.status.busy": "2026-04-15T12:05:30.666220Z", "iopub.status.idle": "2026-04-15T12:05:30.668704Z", "shell.execute_reply": "2026-04-15T12:05:30.668117Z"}
# Tau settings in seconds
tau_start = 1e-6  # s
tau_stop = 500e-6  # s
tau_step = 10e-6  # s

repetitions = 1000

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:05:30.670154Z", "iopub.status.busy": "2026-04-15T12:05:30.670011Z", "iopub.status.idle": "2026-04-15T12:05:32.627277Z", "shell.execute_reply": "2026-04-15T12:05:32.625895Z"}
t1_sched = Schedule(name="t1_experiment")

with (
    t1_sched.loop(arange(0, repetitions, 1, DType.NUMBER)),
    t1_sched.loop(arange(start=tau_start, stop=tau_stop, step=tau_step, dtype=DType.TIME)) as tau,
):
    t1_sched.add(Reset(qubit.name))
    # Prepare |1>
    t1_sched.add(X(qubit=qubit.name))
    # Measure after time tau
    t1_sched.add(Measure(qubit.name, coords={"tau": tau}, acq_channel="S_21"), rel_time=tau)

# Execute the experiment
t1_data = hw_agent.run(t1_sched)
if cluster.is_dummy:
    example_data = open_dataset("./dependencies/datasets/t1.hdf5", engine="h5netcdf")
    t1_data = t1_data.update({"S_21": example_data.S_21})

# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:05:32.632678Z", "iopub.status.busy": "2026-04-15T12:05:32.632391Z", "iopub.status.idle": "2026-04-15T12:05:34.103317Z", "shell.execute_reply": "2026-04-15T12:05:34.101113Z"}
t1_analysis = T1Analysis(t1_data).run()
t1_analysis.display_figs_mpl()

# %% [markdown]
# # Ramsey
# In a Ramsey experiment, the qubit is placed in a superposition of |0⟩ and |1⟩ using
# an $X_{\pi/2}$ pulse, while a frequency offset is given to its port clock, such that the qubit
# accumulates a phase over time as it moves along the Bloch sphere equator. After some
# time $\tau$, a second $X_{\pi/2}$ pulse is played such that the qubit is moved either towards
# the |0⟩ or the |1⟩ state depending on its phase. By sweeping $\tau$, an oscillation will be
# observed corresponding to the qubit's detuning from its "true" $f_{01}$ frequency, with a
# decay corresponding to the qubit's $T_2^*$.

# %% [markdown]
# ## Experiment settings
# %% execution={"iopub.execute_input": "2026-04-15T12:05:34.107032Z", "iopub.status.busy": "2026-04-15T12:05:34.106678Z", "iopub.status.idle": "2026-04-15T12:05:34.113078Z", "shell.execute_reply": "2026-04-15T12:05:34.112001Z"}
# Detuning
frequency_detuning = 1e6  # Hz

# Tau settings in seconds
tau_start = 1e-6  # s
tau_stop = 25e-6  # s
tau_step = 0.5e-6  # s

repetitions = 1000

# %% [markdown]
# ## Experiment schedule
# Note: in order for the second ${\pi/2}$ pulse to have a phase difference $\Delta \phi = \Delta f \tau$, relative to the first $X_{\pi/2}$ pulse we offset the clock frequency by $\Delta f$ for a time $\tau$.
# Here, $\Delta f$ is the `frequency_detuning` and $\tau$ is `tau`.
# %% execution={"iopub.execute_input": "2026-04-15T12:05:34.116489Z", "iopub.status.busy": "2026-04-15T12:05:34.116194Z", "iopub.status.idle": "2026-04-15T12:05:36.707755Z", "shell.execute_reply": "2026-04-15T12:05:36.706286Z"}
ramsey_sched = Schedule(name="ramsey_experiment")
with (
    ramsey_sched.loop(arange(0, repetitions, 1, DType.NUMBER)),
    ramsey_sched.loop(
        arange(start=tau_start, stop=tau_stop, step=tau_step, dtype=DType.TIME)
    ) as tau,
):
    ramsey_sched.add(Reset(qubit.name))
    # Play X/2 pulse
    ramsey_sched.add(X90(qubit=qubit.name))

    # Implementing a phase kick:
    # Detune the qubit's clock by the required frequency detuning
    ramsey_sched.add(
        SetClockFrequency(
            clock=f"{qubit.name}.01",
            clock_freq_new=qubit.clock_freqs.f01 + frequency_detuning,
        )
    )
    # After a time tau, reset the qubit clock frequency to its original value
    ramsey_sched.add(
        SetClockFrequency(
            clock=f"{qubit.name}.01",
            clock_freq_new=qubit.clock_freqs.f01,
        ),
        rel_time=tau,
    )

    # Play second X/2 pulse after time tau
    ramsey_sched.add(X90(qubit=qubit.name))
    ramsey_sched.add(Measure(qubit.name, coords={"tau": tau}, acq_channel="S_21"))

# Execute the experiment
ramsey_data = hw_agent.run(ramsey_sched)
if cluster.is_dummy:
    example_data = open_dataset("./dependencies/datasets/ramsey.hdf5", engine="h5netcdf")
    ramsey_data = ramsey_data.update({"S_21": example_data.S_21})
# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:05:36.711411Z", "iopub.status.busy": "2026-04-15T12:05:36.711149Z", "iopub.status.idle": "2026-04-15T12:05:38.353624Z", "shell.execute_reply": "2026-04-15T12:05:38.351038Z"}
ramsey_analysis = RamseyAnalysis(ramsey_data).run()
ramsey_analysis.display_figs_mpl()

# %% [markdown]
# # Echo
# The Hahn echo experiment is a modified Ramsey sequence that uses a refocusing
# $\pi$-pulse at the midpoint of the free evolution time (t = $\tau$/2) to invert the phase evolution,
# which effectively cancels out phase accumulations due to frequency offsets that are constant within the duration of the experiment (low-f noise).
# The fitted decay time $T_{2,e}$ can be compared to $T_2^*$ to estimate low-frequency noise.

# %% [markdown]
# ## Experiment settings
# Note: in order for the second ${\pi/2}$ pulse to have a phase difference $\Delta \phi = \Delta f \tau$, relative to the first $X_{\pi/2}$ pulse we offset the clock frequency by $\Delta f$ for a time $\tau$.
# Here, $\Delta f$ is the `frequency_detuning` and $\tau$ is `tau`.
# %% execution={"iopub.execute_input": "2026-04-15T12:05:38.357785Z", "iopub.status.busy": "2026-04-15T12:05:38.357396Z", "iopub.status.idle": "2026-04-15T12:05:38.364479Z", "shell.execute_reply": "2026-04-15T12:05:38.363004Z"}
# Detuning
frequency_detuning = 1e6  # Hz

# Tau settings
tau_start = 1e-6  # s
tau_stop = 100e-6  # s
tau_step = 2e-6  # s

repetitions = 1000

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:05:38.367414Z", "iopub.status.busy": "2026-04-15T12:05:38.367098Z", "iopub.status.idle": "2026-04-15T12:05:41.577450Z", "shell.execute_reply": "2026-04-15T12:05:41.575303Z"}
echo_sched = Schedule(name="echo_experiment")
echo_sched.add(
    SetClockFrequency(
        clock=qubit.name + ".01", clock_freq_new=qubit.clock_freqs.f01 + frequency_detuning
    )
)
# Update parameters
echo_sched.add(IdlePulse(4e-9))

with (
    echo_sched.loop(arange(0, repetitions, 1, DType.NUMBER)),
    echo_sched.loop(arange(start=tau_start, stop=tau_stop, step=tau_step, dtype=DType.TIME)) as tau,
):
    echo_sched.add(Reset(qubit.name))
    # Play first X/2 pulse
    echo_sched.add(X90(qubit=qubit.name))
    # Add reflecting pi pulse at time tau / 2
    echo_sched.add(X(qubit=qubit.name), rel_time=tau / 2)
    # Play second X/2 pulse tau / 2 seconds after pi pulse
    echo_sched.add(X90(qubit=qubit.name), rel_time=tau / 2)
    echo_sched.add(Measure(qubit.name, coords={"tau": tau}, acq_channel="S_21"))

# Execute the experiment
echo_data = hw_agent.run(echo_sched)
if cluster.is_dummy:
    example_data = open_dataset("./dependencies/datasets/echo.hdf5", engine="h5netcdf")
    echo_data = echo_data.update({"S_21": example_data.S_21})

# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:05:41.581783Z", "iopub.status.busy": "2026-04-15T12:05:41.581424Z", "iopub.status.idle": "2026-04-15T12:05:43.141989Z", "shell.execute_reply": "2026-04-15T12:05:43.140479Z"}
echo_analysis = EchoAnalysis(echo_data).run()
echo_analysis.display_figs_mpl()

# %% [markdown]
# # Single shot readout
# This experiment is performed to calibrate single-shot readout for transmon qubits. The qubit is
# prepared in either the |0⟩ or |1⟩ state, after which the resonator response is plotted in
# the IQ plane. From the position of the two centroids a discriminator line is drawn that
# can be used on the FPGA to classify single shots of the readout resonator into the |0⟩
# and |1⟩ qubit states. Furthermore, the structure of the two centroids allows us to quantify
# how well the two states can be distinguished and to find the State Preparation and Measurement (SPAM)
# errors. This experiment allows the user to determine the appropriate acquisition rotation and threshold,
# as described on the page documenting [readout](../../../products/architecture/sequencers/readout.md).

# %% [markdown]
# ## Experiment settings

# %% execution={"iopub.execute_input": "2026-04-15T12:05:43.146490Z", "iopub.status.busy": "2026-04-15T12:05:43.146113Z", "iopub.status.idle": "2026-04-15T12:05:43.152378Z", "shell.execute_reply": "2026-04-15T12:05:43.151074Z"}
num_shots = 1000

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:05:43.155540Z", "iopub.status.busy": "2026-04-15T12:05:43.155204Z", "iopub.status.idle": "2026-04-15T12:05:55.532842Z", "shell.execute_reply": "2026-04-15T12:05:55.531349Z"}
ssro_sched = Schedule("Readout")

with ssro_sched.loop(arange(start=0, stop=num_shots, step=1, dtype=DType.NUMBER)) as rep:
    ssro_sched.add(Reset(qubit.name))
    # Measure |0>
    ssro_sched.add(Measure(qubit.name, coords={"reps": rep, "state": 0}, acq_channel="S_21"))
    # Prepare |1>
    ssro_sched.add(Reset(qubit.name))
    ssro_sched.add(X(qubit=qubit.name))
    # Measure |1>
    ssro_sched.add(Measure(qubit.name, coords={"reps": rep, "state": 1}, acq_channel="S_21"))

# Execute the experiment
ssro_data = hw_agent.run(ssro_sched)
if cluster.is_dummy:
    example_data = open_dataset(
        "./dependencies/datasets/single_shot_readout.hdf5", engine="h5netcdf"
    )
    ssro_data = ssro_data.update({"S_21": example_data.S_21})


# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:05:55.536208Z", "iopub.status.busy": "2026-04-15T12:05:55.535804Z", "iopub.status.idle": "2026-04-15T12:05:57.212787Z", "shell.execute_reply": "2026-04-15T12:05:57.211891Z"}
ssro_analysis = SSROAnalysis(ssro_data).run()
ssro_analysis.display_figs_mpl()

# %% [markdown]
# ## Post-run
# %% execution={"iopub.execute_input": "2026-04-15T12:05:57.217052Z", "iopub.status.busy": "2026-04-15T12:05:57.216864Z", "iopub.status.idle": "2026-04-15T12:05:57.220503Z", "shell.execute_reply": "2026-04-15T12:05:57.219631Z"}
# Update device config
qubit.measure.acq_rotation = ssro_analysis.quantities_of_interest["acq_rotation_rad"].nominal_value
qubit.measure.acq_threshold = ssro_analysis.quantities_of_interest["acq_threshold"].nominal_value

# %% [markdown]
# # Single qubit randomized benchmarking
# Single-qubit randomized benchmarking measures the single-qubit gates fidelity.
# For this experiment a sequence of random Clifford gates is generated (one sequence per seed),
# and for each experiment an increasing portion of this sequence is executed (different gate
# string lengths m) before the qubit is returned to its initial state. A fit of the exponential
# decay of the overlap between initial and final states gives the average error per Clifford
# gate, while the variance in this $⟨\psi_f|\psi_i⟩$ overlap between seeds for the same gate string
# length gives an estimate of the degree to which this error is coherent.

# %% [markdown]
# ## Experiment settings
# %% execution={"iopub.execute_input": "2026-04-15T12:05:57.222695Z", "iopub.status.busy": "2026-04-15T12:05:57.222463Z", "iopub.status.idle": "2026-04-15T12:05:57.226277Z", "shell.execute_reply": "2026-04-15T12:05:57.225372Z"}
# RB settings
lengths = np.arange(0, 40, 5)
seeds = np.random.randint(0, 2**31 - 1, size=10, dtype=np.int32)

repetitions = 1

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:05:57.229021Z", "iopub.status.busy": "2026-04-15T12:05:57.228786Z", "iopub.status.idle": "2026-04-15T12:06:01.587658Z", "shell.execute_reply": "2026-04-15T12:06:01.586754Z"}
sched = randomized_benchmarking_schedule(
    qubit.name,
    lengths=lengths,
    repetitions=repetitions,
    seeds=seeds,
)

# Execute the experiment
rb_data = hw_agent.run(sched)
if cluster.is_dummy:
    example_data = open_dataset(
        "./dependencies/datasets/single_qubit_randomized_benchmarking.hdf5", engine="h5netcdf"
    )
    rb_data.update({"S_21": example_data.S_21, "calibration": example_data.calibration})

# %% [markdown]
# ## Analyze the experiment
# %% execution={"iopub.execute_input": "2026-04-15T12:06:01.590804Z", "iopub.status.busy": "2026-04-15T12:06:01.590608Z", "iopub.status.idle": "2026-04-15T12:06:01.753392Z", "shell.execute_reply": "2026-04-15T12:06:01.752539Z"}
rb_analysis = RBAnalysis(rb_data).run()
rb_analysis.display_figs_mpl()

# %% [markdown]
# # Two-qubit randomized benchmarking
# Two-qubit randomized benchmarking characterizes the overall Clifford (in)fidelity, which includes contributions
# from both single- and two-qubit gates.
# For this experiment a random sequence of two-qubit Cliffords is generated (one sequence per seed),
# and for each experiment an increasing portion of this sequence is executed (different gate
# string lengths $m$) before the two qubits are returned to their initial states. A fit of the exponential
# decay of the overlap between initial and final states gives the average error per gate,
# while the variance in this $⟨\psi_f|\psi_i⟩$ overlap between seeds for the same gate string
# length gives an estimate of the degree to which this error is coherent.

# %% [markdown]
# ## Experiment settings
# %% execution={"iopub.execute_input": "2026-04-15T12:06:01.755810Z", "iopub.status.busy": "2026-04-15T12:06:01.755646Z", "iopub.status.idle": "2026-04-15T12:06:01.759032Z", "shell.execute_reply": "2026-04-15T12:06:01.758300Z"}
# RB settings
lengths = [1]  # number
seeds = np.random.randint(0, 2**31 - 1, size=1, dtype=np.int32)  # number
repetitions = 1

# %% [markdown]
# ## Experiment schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:06:01.760815Z", "iopub.status.busy": "2026-04-15T12:06:01.760642Z", "iopub.status.idle": "2026-04-15T12:06:01.812290Z", "shell.execute_reply": "2026-04-15T12:06:01.811329Z"}
sched = randomized_benchmarking_schedule(
    [q0.name, q2.name],
    lengths=lengths,
    repetitions=repetitions,
    seeds=seeds,
    generator=TwoQubitCliffordCZ,  # Change to the appropriate gate for your architecture!
)

# Compile the schedule
comp_sched = hw_agent.compile(sched)

# %% [markdown]
# ## Show the schedule
# %% execution={"iopub.execute_input": "2026-04-15T12:06:01.816150Z", "iopub.status.busy": "2026-04-15T12:06:01.815975Z", "iopub.status.idle": "2026-04-15T12:06:02.030652Z", "shell.execute_reply": "2026-04-15T12:06:02.029924Z"}
fig, ax = comp_sched.plot_pulse_diagram(plot_backend="mpl")
ax.set_xlim(q2.reset.duration, q2.reset.duration + 2e-6)
plt.show()

# %% [markdown] tags=["footer_1"]
# #### Update the device configuration file
# After measurement, we may store the measured device properties inside a new file to use in future experiments.
# The time-unique identifier ensures that it is easy to find back previously found measurement results.
# %% execution={"iopub.execute_input": "2026-04-15T12:06:02.033526Z", "iopub.status.busy": "2026-04-15T12:06:02.033346Z", "iopub.status.idle": "2026-04-15T12:06:02.037943Z", "shell.execute_reply": "2026-04-15T12:06:02.037292Z"} tags=["footer_2"]
hw_agent.quantum_device.to_json_file("./dependencies/configs", add_timestamp=True)
