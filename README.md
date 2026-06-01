# Qblox Measurement Nodes

Reusable measurement and calibration nodes for superconducting qubit experiments using Qblox hardware.

## Goal

This repository separates:

1. hardware configuration
2. device configuration
3. schedule construction
4. experiment execution
5. data analysis
6. calibration workflow

## Calibration Flow

0. Time of flight
1. Resonator spectroscopy full bandwidth
2. Resonator spectroscopy fine scan
3. Resonator punchout
4. Resonator flux spectroscopy
5. Compensated flux spectroscopy
6. Qubit spectroscopy
7. Rabi
8. Ramsey
9. T1
10. Readout optimization

## Repository Structure

```text
configs/
src/qblox_lab/
notebooks/
scripts/
tests/
