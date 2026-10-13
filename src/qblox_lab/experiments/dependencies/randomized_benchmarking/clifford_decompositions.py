"""
5 primitives decomposition of the single qubit clifford group as per Asaad et al. arXiv:1508.06676.

In this decomposition the Clifford group is represented by 5 primitive gates
that are consecutively applied. (Note that X90 occurs twice in this list).
-X90-Y90-X90-mX180-mY180-

Note: now that I think some more about it this way of representing the 5
primitives decomposition may not be the most useful one.
"""

from __future__ import annotations

five_primitives_decomposition: dict[int, list[str]] = {}
# In the lists below, operators are ordered in time, i.e. the order in which operators should be applied to qubits so that to realize a given Clifford net gate.
# That means that the Clifford gates are equal to the matrix product of the listed operators in REVERSED order.
five_primitives_decomposition[0] = ["I"]
five_primitives_decomposition[1] = ["Y90", "X90"]
five_primitives_decomposition[2] = ["X90", "Y90", "mX180"]
five_primitives_decomposition[3] = ["mX180"]
five_primitives_decomposition[4] = ["Y90", "X90", "mY180"]
five_primitives_decomposition[5] = ["X90", "Y90", "mY180"]
five_primitives_decomposition[6] = ["mY180"]
five_primitives_decomposition[7] = ["Y90", "X90", "mX180", "mY180"]
five_primitives_decomposition[8] = ["X90", "Y90"]
five_primitives_decomposition[9] = ["mX180", "mY180"]
five_primitives_decomposition[10] = ["Y90", "X90", "mX180"]
five_primitives_decomposition[11] = ["X90", "Y90", "mX180", "mY180"]

five_primitives_decomposition[12] = ["Y90", "mX180"]
five_primitives_decomposition[13] = ["X90", "mX180"]
five_primitives_decomposition[14] = ["X90", "Y90", "X90", "mY180"]
five_primitives_decomposition[15] = ["Y90", "mY180"]
five_primitives_decomposition[16] = ["X90"]
five_primitives_decomposition[17] = ["X90", "Y90", "X90"]
five_primitives_decomposition[18] = ["Y90", "mX180", "mY180"]
five_primitives_decomposition[19] = ["X90", "mY180"]
five_primitives_decomposition[20] = ["X90", "Y90", "X90", "mX180", "mY180"]
five_primitives_decomposition[21] = ["Y90"]
five_primitives_decomposition[22] = ["X90", "mX180", "mY180"]
five_primitives_decomposition[23] = ["X90", "Y90", "X90", "mX180"]

"""
Gate decomposition decomposition of the clifford group as per
Epstein et al. Phys. Rev. A 89, 062321 (2014)
"""
epstein_efficient_decomposition: dict[int, list[str]] = {}
# In the lists below, operators are ordered in time, i.e. the order in which operators should be applied to qubits so that to realize a given Clifford net gate.
# That means that the Clifford gates are equal to the matrix product of the listed operators in REVERSED order.

epstein_efficient_decomposition[0] = ["I"]
epstein_efficient_decomposition[1] = ["Y90", "X90"]
epstein_efficient_decomposition[2] = ["mX90", "mY90"]
epstein_efficient_decomposition[3] = ["X180"]
epstein_efficient_decomposition[4] = ["mY90", "mX90"]
epstein_efficient_decomposition[5] = ["X90", "mY90"]
epstein_efficient_decomposition[6] = ["Y180"]
epstein_efficient_decomposition[7] = ["mY90", "X90"]
epstein_efficient_decomposition[8] = ["X90", "Y90"]
epstein_efficient_decomposition[9] = ["X180", "Y180"]
epstein_efficient_decomposition[10] = ["Y90", "mX90"]
epstein_efficient_decomposition[11] = ["mX90", "Y90"]

epstein_efficient_decomposition[12] = ["Y90", "X180"]
epstein_efficient_decomposition[13] = ["mX90"]
epstein_efficient_decomposition[14] = ["X90", "mY90", "mX90"]
epstein_efficient_decomposition[15] = ["mY90"]
epstein_efficient_decomposition[16] = ["X90"]
epstein_efficient_decomposition[17] = ["X90", "Y90", "X90"]
epstein_efficient_decomposition[18] = ["mY90", "X180"]
epstein_efficient_decomposition[19] = ["X90", "Y180"]
epstein_efficient_decomposition[20] = ["X90", "mY90", "X90"]
epstein_efficient_decomposition[21] = ["Y90"]
epstein_efficient_decomposition[22] = ["mX90", "Y180"]
epstein_efficient_decomposition[23] = ["X90", "Y90", "mX90"]
