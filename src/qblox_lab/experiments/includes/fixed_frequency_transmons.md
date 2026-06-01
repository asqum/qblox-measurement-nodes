# Fixed-frequency transmon

## Architecture

```{figure} /applications/superconducting/figures/fixed_frequency_layout.svg
---
width: 700px
align: center
---
Typical Qblox control electronics setup for a QPU of five fixed-frequency transmons.
```

Architectures with fixed-frequency qubits and couplers save on the number of wires inside the fridge, as there are no
flux lines present. Consequently this approach reduces the hardware cost and complexity and simplifies the tuneup, at
the price of more complex single- and two-qubit gate implementations caused by the always-on coupling between
neighboring qubits.

```{admonition} Getting started
---
class: hint
---
A startup guide for fixed-frequency transmon qubits can be found [here](/applications/superconducting/fixed_frequency_transmon/000_transmon_setup.ipynb).
```

```{admonition} Full application guide
---
class: seealso
---
A notebook containing all experiment for this architecture can be found [here](/applications/superconducting/fixed_frequency_transmon/fixed_frequency_transmon_tuneup.ipynb).
```

## Tuneup

```{nblinkgallery}
/applications/superconducting/fixed_frequency_transmon/010_time_of_flight.ipynb
/applications/superconducting/fixed_frequency_transmon/020_resonator_spectroscopy.ipynb
/applications/superconducting/fixed_frequency_transmon/030_resonator_punchout.ipynb
/applications/superconducting/fixed_frequency_transmon/050_qubit_spectroscopy.ipynb
/applications/superconducting/fixed_frequency_transmon/070_rabi.ipynb
/applications/superconducting/fixed_frequency_transmon/110_single_shot_readout.ipynb
```

## Benchmarking

```{nblinkgallery}
/applications/superconducting/fixed_frequency_transmon/080_t1.ipynb
/applications/superconducting/fixed_frequency_transmon/090_ramsey.ipynb
/applications/superconducting/fixed_frequency_transmon/100_echo.ipynb
/applications/superconducting/fixed_frequency_transmon/300_randomized_benchmarking.ipynb
/applications/superconducting/fixed_frequency_transmon/400_two_qubit_randomized_benchmarking.ipynb
```

```{toctree}
---
hidden:
glob:
---
/applications/superconducting/fixed_frequency_transmon/*

```
