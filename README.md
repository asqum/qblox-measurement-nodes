# Qblox Measurement Nodes

A minimal repository for single-qubit calibration with Qblox Scheduler. It includes
readout time-of-flight calibration, standard and segmented broadband resonator
spectroscopy, resonator punchout, resonator flux spectroscopy, and segmented
broadband and power-dependent qubit spectroscopy, Rabi calibration, and qubit
energy-relaxation, dephasing, pulse-shape, gate-diagnostic, and randomized-
benchmarking measurements.

The code uses public scheduler interfaces: `HardwareAgent`, `Schedule`, operations,
`SetParameter`, `SetHardwareOption`, `ResonatorModel`,
`ResonatorFluxSpectroscopyAnalysis`, and `QubitSpectroscopyAnalysis`. It does not
import `qblox_instruments`.

## Structure

```text
config/
  hw_config.json                     Hardware topology and connectivity
  dut_config.json                    Scheduler quantum-device parameters
  flux_config.json                   Persistent logical flux biases
src/qblox_lab/
  config/
    hardware.py                       HardwareAgent and flux-bias handling
    device.py                         Device configuration saving
  experiments/
    cal01_time_of_flight.py            TimeOfFlight experiment
    cal02_resonator_spectroscopy_full_bandwidth.py
                                      BroadbandResonatorSpectroscopy experiment
    cal03_resonator_spectroscopy.py   ResonatorSpectroscopy experiment
    cal04_resonator_punchout.py       ResonatorPunchout experiment
    cal05_resonator_flux_spectroscopy.py
                                      ResonatorFluxSpectroscopy experiment
    cal06_qubit_spectroscopy_full_bandwidth.py
                                      BroadbandQubitSpectroscopy experiment
    cal07_qubit_spectroscopy_intrinsic_width.py
                                      QubitSpectroscopyIntrinsicWidth experiment
    cal08_rabi.py                     Rabi experiment
    cal09_energy_relaxation.py        EnergyRelaxation experiment
    cal10_ramsey.py                   Ramsey experiment
    cal11_echo.py                     Echo experiment
    cal12_drag.py                     DragCalibration experiment
    cal13_allxy.py                    AllXY experiment
    cal14_single_qubit_randomized_benchmarking.py
                                      SingleQubitRandomizedBenchmarking experiment
notebooks/
  run_time_of_flight.ipynb             Time-of-flight run parameters and execution
  run_broadband_resonator_spectroscopy.ipynb
                                      Broadband run parameters and execution
  run_resonator_spectroscopy.ipynb    Run parameters and execution
  run_resonator_punchout.ipynb        Punchout run parameters and execution
  run_resonator_flux_spectroscopy.ipynb
                                      Flux-scan run parameters and execution
  run_broadband_qubit_spectroscopy.ipynb
                                      Broadband qubit scan and analysis
  run07_qubit_spectroscopy_intrinsic_width.ipynb
                                      Power-dependent qubit scan and analysis
  run08_rabi.ipynb                    Amplitude/time Rabi calibration
  run09_energy_relaxation.ipynb       Qubit T1 measurement and analysis
  run10_ramsey.ipynb                  Ramsey T2* and frequency refinement
  run11_echo.ipynb                    Hahn-echo T2 measurement
  run12_drag.ipynb                    DRAG beta calibration
  run13_allxy.ipynb                   AllXY gate diagnostic
  run14_single_qubit_randomized_benchmarking.ipynb
                                      Single-qubit Clifford RB
```

The Python modules contain reusable experiment and configuration logic. All values
that normally change between measurements are defined in the notebook.

## Installation

Activate the environment containing Qblox Scheduler, then install this repository:

```powershell
C:\Users\yi.huang\anaconda3\envs\qse\python.exe -m pip install -e .
```

The repository pins `qblox-scheduler==1.0.0b8`. Scheduler 1.0.1 moved the
superconducting `BasicTransmonElement` into a separate applications distribution;
without that distribution it cannot load this repository's device configuration.

Open [run_resonator_spectroscopy.ipynb](notebooks/run_resonator_spectroscopy.ipynb)
with the `qse` Jupyter kernel.

For acquisition-delay calibration, open
[run_time_of_flight.ipynb](notebooks/run_time_of_flight.ipynb). It acquires a named
complex trace using `Measure(..., acq_protocol="Trace")`, converts it to magnitude,
and delegates threshold detection to Qblox Scheduler's public `TimeOfFlightAnalysis`.
Successful results can update the device's `measure.acq_delay`. The NCO propagation
delay is reported but is not applied through a private instrument interface.

For a two-dimensional readout-power scan, open
[run_resonator_punchout.ipynb](notebooks/run_resonator_punchout.ipynb). The experiment
sweeps `Measure.pulse_amp` and frequency through scheduler loop variables and returns
one dataset with named amplitude and frequency coordinates. Analysis averages repeated
samples onto a complex grid and tracks the minimum-transmission frequency at each
amplitude. Selecting and saving a final readout amplitude remains an explicit user
choice.

The cal01-cal04 and cal06-cal14 experiment constructors accept an optional
`flux_config`. When it is supplied, the selected qubits are ramped to their stored
biases immediately before measurement. When it is `None`, those experiments do not
resolve, read, set, or restore any flux parameter, so the live hardware state is
left unchanged.

For a two-dimensional resonator-versus-flux scan, open
[run_resonator_flux_spectroscopy.ipynb](notebooks/run_resonator_flux_spectroscopy.ipynb).
The hardware helper resolves each logical flux port through the connectivity graph,
then the experiment always sweeps that parameter through the scheduler's public
`SetParameter`. If `flux_config` is supplied, its ramp settings are used and its
stored bias is restored after the sweep, including when execution fails. If it is
omitted, default ramp settings are used and the output is restored to 0 V. Applying
and saving a fitted sweet spot are separate, explicit notebook actions. The fit and
figures use the public
`ResonatorFluxSpectroscopyAnalysis`; simulation uses the scheduler's cosine and
complex hanger functions.

For a sweep wider than 1 GHz, open
[run_broadband_resonator_spectroscopy.ipynb](notebooks/run_broadband_resonator_spectroscopy.ipynb).
The broadband experiment divides the requested frequency grid into contiguous branches
no wider than 800 MHz. It sets each branch LO to the branch midpoint, executes all
branches through one top-level `HardwareAgent.run(...)`, and returns the scheduler's
single combined dataset. A final hardware step restores the configured LO; a fallback
restoration is also attempted if acquisition fails.

For a broadband qubit-frequency search, open
[run_broadband_qubit_spectroscopy.ipynb](notebooks/run_broadband_qubit_spectroscopy.ipynb).
The cal06 experiment applies a square spectroscopy pulse before readout and divides
the requested drive-frequency grid into branches no wider than 800 MHz. Each drive LO
is placed at its branch midpoint, all branches are acquired in one top-level run, and
the configured drive LO is restored afterward. Repetitions are averaged before the
public `QubitSpectroscopyAnalysis` performs a Lorentzian fit. Simulated data uses the
same public Lorentzian function with configurable transition frequency, linewidth,
contrast, phase, noise, and random seed. A successful result can update the in-memory
device's `clock_freqs.f01` before saving to a new device file.

For estimating the low-power qubit linewidth, open
[run07_qubit_spectroscopy_intrinsic_width.ipynb](notebooks/run07_qubit_spectroscopy_intrinsic_width.ipynb).
The cal07 experiment repeats the cal06 frequency sweep for each requested normalized
drive amplitude and returns one dataset with frequency and drive-amplitude coordinates.
The user selects one acquired amplitude for analysis; cal07 then delegates the selected
trace to cal06's public `QubitSpectroscopyAnalysis` workflow. The complete multi-power
dataset remains available after fitting.

For pi-pulse calibration, open [run08_rabi.ipynb](notebooks/run08_rabi.ipynb).
The cal08 experiment accepts an evenly spaced amplitude array, duration array, or
both. It uses real-time scheduler loops and returns one dataset containing named
amplitude and duration coordinates. Its data plot is one-dimensional when one axis
is fixed and two-dimensional when both axes are swept. Power Rabi analysis fits the
amplitude trace at a user-selected duration; time Rabi analysis fits the duration
trace at a user-selected amplitude. Both fits use Qblox Scheduler's public
`RabiAnalysis`, and simulation uses its public cosine function. A successful result
can update the matching `rxy.amp180` and `rxy.duration` pair in a new device file.

For qubit energy-relaxation measurement, open
[run09_energy_relaxation.ipynb](notebooks/run09_energy_relaxation.ipynb). The cal09
experiment applies the device's calibrated `X` operation, waits for a real-time
swept delay, and measures the remaining excited-state signal. Repetitions are
averaged before Qblox Scheduler's public `T1Analysis` fits the exponential decay.
Simulated data uses the same public `exp_decay_func`, with configurable T1,
baseline, contrast, stretch factor, phase, noise, and seed. Since the scheduler's
`BasicTransmonElement` has no T1 field, the fitted value is reported in the result
object and is not written into the device configuration.

For Ramsey frequency refinement and dephasing, open
[run10_ramsey.ipynb](notebooks/run10_ramsey.ipynb). The real-time schedule advances
the second pi/2 pulse phase according to the artificial detuning. Public
`RamseyAnalysis` extracts `T2*`, the physical detuning, and a refined `f01`; only
the refined frequency can be applied to a new device configuration. Simulation uses
the same public damped-oscillation model.

For Hahn-echo dephasing, open [run11_echo.ipynb](notebooks/run11_echo.ipynb). The
schedule places an X pulse halfway between two X90 pulses and sweeps their total
separation. Public `EchoAnalysis` extracts `T2 echo`, and simulation uses its
exponential-decay model. The result is reported but not persisted because the device
schema has no coherence-time field.

For pulse-shape calibration, open [run12_drag.ipynb](notebooks/run12_drag.ipynb).
The node measures the `X-Y90` and `Y-X90` sequences across a real-time DRAG-beta
sweep. Their projected linear-response intersection determines `rxy.beta`, which can
be saved to a new device file. An optional experiment-scoped drive amplitude leaves
waveform headroom for the derivative quadrature; when it is reduced, drive attenuation
can be adjusted to preserve the intended physical rotation.

For the standard 21-pair gate diagnostic, open
[run13_allxy.ipynb](notebooks/run13_allxy.ipynb). Public Scheduler `Rxy` operations
construct the same sequence set used by `allxy_sched`, and public `AllXYAnalysis`
normalizes the IQ data, plots the ideal staircase, and reports its mean deviation.
AllXY is diagnostic and does not automatically change a device parameter.

For reference single-qubit Clifford randomized benchmarking, open
[run14_single_qubit_randomized_benchmarking.ipynb](notebooks/run14_single_qubit_randomized_benchmarking.ipynb).
Cal14 generates the 24 Clifford elements from public `Rxy` gates, appends the exact
recovery Clifford, acquires all lengths and random circuits in one top-level run, and
fits `A p**m + B`. It reports the average Clifford fidelity, error per Clifford, and
an estimated error per primitive pulse. The random seed makes the circuit set
reproducible. RB is a performance measurement and does not mutate the device.

The broadband resonator experiment can also generate data by calling
`dataset = experiment.simulated_data()`. Its simulation uses Qblox Scheduler's public
complex hanger model with configurable resonance frequency, loaded and coupling
quality factors, signal amplitude, electrical delay, noise, and random seed. Omitted
resonance frequencies come from the device configuration; unavailable quality factors
use typical values.

## Notebook workflow

The notebook is organized into five operations:

1. Import the hardware helper, device helper, and experiment class.
2. Define configuration paths and every run parameter in one cell.
3. Create `HardwareAgent` and `ResonatorSpectroscopy`.
4. Call `run_measurement(...)` with all measurement parameters.
5. Call `analysis()`, optionally plot, and optionally save fitted frequencies.

The central measurement call is:

```python
dataset = experiment.run_measurement(
    frequency_width=FREQUENCY_WIDTH,
    frequency_points=FREQUENCY_POINTS,
    repetitions=REPETITIONS,
    readout_amplitude=READOUT_AMPLITUDE,
    output_attenuation=OUTPUT_ATTENUATION,
    input_attenuation=INPUT_ATTENUATION,
    readout_lo_frequency=READOUT_LO_FREQUENCY,
    timeout=TIMEOUT,
)
```

Analysis is deliberately separate:

```python
results = experiment.analysis()
```

The scan for each qubit is centered on `clock_freqs.readout` from the supplied device
configuration. Readout amplitude, attenuation, and LO overrides are experiment-scoped
scheduler steps and are not written to the input configuration.

To apply successful fitted resonance frequencies, set `SAVE_FITTED_DEVICE` to an output
path in the notebook. Leaving it as `None` keeps the device configuration unchanged.
