# Qblox Measurement Nodes

A minimal repository for single-qubit calibration with Qblox Scheduler. It includes
readout time-of-flight calibration, standard and segmented broadband resonator
spectroscopy, resonator punchout, resonator flux spectroscopy, and segmented
broadband and power-dependent qubit spectroscopy.

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
```

The Python modules contain reusable experiment and configuration logic. All values
that normally change between measurements are defined in the notebook.

## Installation

Activate the environment containing Qblox Scheduler, then install this repository:

```powershell
C:\Users\yi.huang\anaconda3\envs\qse\python.exe -m pip install -e . --no-deps
```

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

The cal01, cal02, cal03, cal04, cal06, and cal07 experiment constructors accept an
optional `flux_config`. When it is supplied, the selected qubits are ramped to their
stored biases immediately before measurement. When it is `None`, those experiments do
not resolve, read, set, or restore any flux parameter, so the live hardware state is
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
