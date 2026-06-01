# Superconducting qubits

On this page, we provide a number of turnkey experiments for different architecture implementations of superconducting
hardware. Qblox offers a range of products that are designed specifically for this type of QPU architecture. Throughout
the experiments, we use the [Cluster](../../products/architecture/cluster/rf_cluster) for microwave readout
([QRM-RF](../../products/architecture/modules/qrm_rf.md)), microwave control
([QCM-RF](../../products/architecture/modules/qcm_rf.md)) and flux control (direct
[QCM](../../products/architecture/modules/qcm.md)).

## Transmons

Transmons are in general controlled with 4-8GHz pulses on qubit/coupler drive lines, and baseband (\<400MHz) pulses on
top of a DC offset on flux lines. These qubits are coupled to resonators typically in the 5-10GHz range, which can be
read out through a shared feedline with pulses on the order of 100ns-10$\mu$s.

`````{tab-set}
````{tab-item} Fixed-frequency transmons
```{include} includes/fixed_frequency_transmons.md
:start-after: "# Fixed-frequency transmon"
:end-before: ```{toctree}

```
````
````{tab-item} Flux-tunable transmons
```{include} includes/flux_tunable_transmons.md
:start-after: "# Flux-tunable transmon"
:end-before: ```{toctree}

```
````
`````

```{toctree}
---
hidden:
maxdepth: 1
---
includes/fixed_frequency_transmons.md
includes/flux_tunable_transmons.md

```
