# Flux-tunable transmon

## Architecture

```{figure} /applications/superconducting/figures/tunable_qubits_layout.svg
---
width: 700px
align: center
---
Typical Qblox control electronics setup for a QPU of five flux-tunable transmons.
```

In architectures with tunable qubits and fixed couplers, the qubit frequencies can be changed by playing a signal on the
qubit's flux line, effectively tuning the interaction strength between two neighboring qubits. Compared to an
architecture where all qubit and coupler frequencies are fixed, this increases hardware overhead as well as the number
of steps in a tuneup procedure. However, this design helps to mitigate crosstalk as inter-qubit coupling can be
completely turned off when idling.

```{admonition} Getting started
---
class: hint
---
A startup guide for flux-tunable transmon qubits can be found [here](/applications/superconducting/flux_tunable_transmon/000_transmon_setup.ipynb).
```

```{admonition} Full application guide
---
class: seealso
---
A notebook containing all experiment for this architecture can be found [here](/applications/superconducting/flux_tunable_transmon/flux_tunable_transmons_tuneup.ipynb).
```

## Tuneup

```{nblinkgallery}
/applications/superconducting/flux_tunable_transmon/010_time_of_flight.ipynb
/applications/superconducting/flux_tunable_transmon/020_resonator_spectroscopy.ipynb
/applications/superconducting/flux_tunable_transmon/030_resonator_punchout.ipynb
/applications/superconducting/flux_tunable_transmon/040_resonator_flux_spectroscopy.ipynb
/applications/superconducting/flux_tunable_transmon/050_qubit_spectroscopy.ipynb
/applications/superconducting/flux_tunable_transmon/070_rabi.ipynb
/applications/superconducting/flux_tunable_transmon/110_single_shot_readout.ipynb
/applications/superconducting/flux_tunable_transmon/120_cphase_chevron.ipynb
```

## Benchmarking

```{nblinkgallery}
/applications/superconducting/flux_tunable_transmon/080_t1.ipynb
/applications/superconducting/flux_tunable_transmon/090_ramsey.ipynb
/applications/superconducting/flux_tunable_transmon/100_echo.ipynb
/applications/superconducting/flux_tunable_transmon/300_randomized_benchmarking.ipynb
/applications/superconducting/flux_tunable_transmon/400_two_qubit_randomized_benchmarking.ipynb
```

```{toctree}
---
hidden:
glob:
---
/applications/superconducting/flux_tunable_transmon/*

```
