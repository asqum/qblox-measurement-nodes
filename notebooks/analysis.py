# analysis.py
import numpy as np
import pandas as pd
import lmfit
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.stats import norm

def set_attenuation(hw_agent, qubits, port_type, target_att):
    """Dynamically set attenuation for a list of qubits using their explicit ports."""
    if not isinstance(qubits, (list, tuple)):
        qubits = [qubits]
        
    out_att_dict = hw_agent.hardware_configuration.hardware_options.output_att
    
    for q in qubits:
        if port_type == 'ro':
            port_key = f'{q.name}:res-{q.name}.ro'
        elif port_type == '01':
            port_key = f'{q.name}:mw-{q.name}.01'
        else:
            raise ValueError("port_type must be 'ro' or '01'")
            
        if port_key in out_att_dict:
            prior_att = out_att_dict[port_key]
            out_att_dict[port_key] = target_att
            print(f"Attenuation changed from {prior_att} to {target_att} via port {port_key}")
        else:
            print(f"Warning: Port {port_key} not found in hardware config. Skipping.")

def set_input_attenuation(hw_agent, qubits, target_att):
    """
    Dynamically set the input attenuation for a list of qubits.
    Input attenuation is only applicable to readout acquisition ports.
    """
    if not isinstance(qubits, (list, tuple)):
        qubits = [qubits]
        
    hw_options = hw_agent.hardware_configuration.hardware_options
    
    # Ensure the input_att dictionary exists just in case
    if hw_options.input_att is None:
        hw_options.input_att = {}
        
    in_att_dict = hw_options.input_att
    
    for q in qubits:
        # Input attenuation only applies to the readout acquisition path
        port_key = f'{q.name}:res-{q.name}.ro'
            
        if port_key in in_att_dict:
            prior_att = in_att_dict[port_key]
            in_att_dict[port_key] = target_att
            print(f"Input attenuation changed from {prior_att} to {target_att} via port {port_key}")
        else:
            in_att_dict[port_key] = target_att
            print(f"Input attenuation initialized to {target_att} via port {port_key}")

def calculate_electrical_delays(qubits, dataset):
    """
    Calculates the phase compensation slope and intercept for off-resonance data
    from the FIRST provided qubit, and applies it to all qubits.
    """
    if not qubits:
        print("No qubits provided.")
        return {}

    # --- 1. Calculate fit for the FIRST qubit only ---
    ref_q = qubits[0]
    freqs_ref = dataset[f"frequency_{ref_q.name}"].values
    s21_ref = dataset[f"S_21_{ref_q.name}"].values
    
    phase_unwrapped_ref = np.unwrap(np.angle(s21_ref))
    ref_slope, ref_intercept = np.polyfit(freqs_ref, phase_unwrapped_ref, 1)
    ref_delay_ns = (-ref_slope / (2 * np.pi)) * 1e9

    print(f"Reference Electrical Delay (from Qubit {ref_q.name}): {ref_delay_ns:.2f} ns")

    # --- 2. Build dictionary for ALL qubits using reference values ---
    phase_calibrations = {}
    for q in qubits:
        phase_calibrations[q.name] = {
            'slope': ref_slope, 
            'intercept': ref_intercept, 
            'delay_ns': ref_delay_ns,
            'freqs_ref': freqs_ref, # Saved for plotting later
            'phase_unwrapped_ref': phase_unwrapped_ref
        }
    
    return phase_calibrations

def gaussian(x, amp, mu, sigma):
    """Standard Gaussian PDF."""
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gauss(data):
    """Fits a Gaussian to 1D data."""
    counts, bins = np.histogram(data, bins=100, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    p0 = [np.max(counts), np.mean(data), np.std(data)]
    try:
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=p0)
        return popt, bins
    except Exception:
        return p0, bins
    
def fit_resonator_flux_sweetspot(fluxes: np.ndarray, min_freqs: np.ndarray):
    """Fits extracted minimum frequencies vs flux to a sine wave."""
    model = lmfit.models.SineModel()
    
    center_guess = np.mean(min_freqs)
    y_centered = min_freqs - center_guess
    
    params = model.guess(y_centered, x=fluxes)
    result = model.fit(y_centered, x=fluxes, params=params)
    result.params.add('center', value=center_guess, vary=False)
    
    amp = result.params['amplitude'].value
    freq = result.params['frequency'].value
    shift = result.params['shift'].value
    
    target_phase = np.pi / 2 if amp > 0 else -np.pi / 2
    n_cycles = np.arange(-5, 6)
    possible_sweetspots = (target_phase - shift + n_cycles * 2 * np.pi) / freq
    
    best_sweetspot = possible_sweetspots[np.argmin(np.abs(possible_sweetspots))]
    max_freq = center_guess + np.abs(amp)
    
    return result, best_sweetspot, max_freq

def fit_qubit_spectroscopy(frequencies: np.ndarray, signal: np.ndarray):
    """Fits a Lorentzian model (plus a constant offset) to qubit spectroscopy data."""
    lor_mod = lmfit.models.LorentzianModel(prefix='peak_')
    const_mod = lmfit.models.ConstantModel(prefix='const_')
    model = lor_mod + const_mod
    
    offset_guess = np.median(signal)
    peak_idx = np.argmax(np.abs(signal - offset_guess))
    center_guess = frequencies[peak_idx]
    amp_guess = (signal[peak_idx] - offset_guess) * ((frequencies[-1] - frequencies[0]) / 10.0)
    
    params = model.make_params(
        peak_center=center_guess,
        peak_amplitude=amp_guess,
        peak_sigma=(frequencies[-1] - frequencies[0]) / 20.0,
        const_c=offset_guess
    )
    
    result = model.fit(signal, params=params, x=frequencies)
    f01 = result.params['peak_center'].value
    linewidth = result.params['peak_fwhm'].value
    
    return result, f01, linewidth


def rabi_osc_func(x, amplitude, frequency, phase, offset):
    """Equation for a standard Rabi cosine oscillation."""
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

def fit_rabi_oscillation(amplitudes: np.ndarray, signal: np.ndarray):
    """
    Fits a Power Rabi fringe to a cosine wave.
    Returns the lmfit result and the pi-pulse amplitude (amp180).
    """
    model = lmfit.Model(rabi_osc_func)
    
    # 1. Basic guesses
    offset_guess = np.mean(signal)
    amp_guess = (np.max(signal) - np.min(signal)) / 2.0
    
    # 2. Smart frequency guess using FFT
    dt = amplitudes[1] - amplitudes[0]
    fft_freqs = np.fft.fftfreq(len(amplitudes), d=dt)
    fft_vals = np.fft.fft(signal - offset_guess)
    
    # Look only at positive frequencies to find the peak
    pos_mask = fft_freqs > 0
    peak_idx = np.argmax(np.abs(fft_vals[pos_mask]))
    freq_guess = fft_freqs[pos_mask][peak_idx]
    
    # 3. Determine phase guess (Rabi usually starts at an extremum)
    if (signal[0] - offset_guess) < 0:
        phase_guess = np.pi
    else:
        phase_guess = 0.0
    
    # 4. Build parameters and enforce physical boundaries
    params = model.make_params(
        amplitude=amp_guess,
        frequency=freq_guess,
        phase=phase_guess,
        offset=offset_guess
    )
    params['frequency'].min = 0.0
    
    # 5. Run fit
    result = model.fit(signal, params=params, x=amplitudes)
    
    # The pi-pulse amplitude is exactly half the period of the fitted cosine wave
    fitted_freq = result.params['frequency'].value
    amp180 = 1.0 / (2.0 * fitted_freq)
    
    return result, amp180

def damped_osc_func(x, amplitude, frequency, phase, tau, offset):
    """Equation for an exponentially decaying sinusoidal oscillation."""
    return amplitude * np.exp(-x / tau) * np.cos(2 * np.pi * frequency * x + phase) + offset

def fit_ramsey_decay(taus: np.ndarray, signal: np.ndarray):
    """Fits a Ramsey fringe to an exponentially damped oscillation."""
    model = lmfit.Model(damped_osc_func)
    
    offset_guess = np.mean(signal)
    amp_guess = (np.max(signal) - np.min(signal)) / 2.0
    tau_guess = np.max(taus) / 2.0
    
    dt = taus[1] - taus[0]
    fft_freqs = np.fft.fftfreq(len(taus), d=dt)
    fft_vals = np.fft.fft(signal - offset_guess)
    
    pos_mask = fft_freqs > 0
    peak_idx = np.argmax(np.abs(fft_vals[pos_mask]))
    freq_guess = fft_freqs[pos_mask][peak_idx]
    
    params = model.make_params(
        amplitude=amp_guess,
        frequency=freq_guess,
        phase=0.0,
        tau=tau_guess,
        offset=offset_guess
    )
    params['tau'].min = 0.0
    params['frequency'].min = 0.0
    
    result = model.fit(signal, params=params, x=taus)
    t2_star = result.params['tau'].value
    detuning = result.params['frequency'].value
    
    return result, t2_star, detuning


def exp_decay_func(x, amplitude, tau, offset):
    """Equation for a standard exponential decay."""
    return amplitude * np.exp(-x / tau) + offset

def fit_exp_decay(x_data: np.ndarray, y_data: np.ndarray):
    """
    Fits 1D data to an exponential decay.
    Returns the lmfit result and the decay time constant (tau).
    """
    model = lmfit.Model(exp_decay_func)
    
    # Smart guesses based on typical T1/Echo shapes
    offset_guess = np.mean(y_data[-10:]) # The tail of the data
    amp_guess = y_data[0] - offset_guess # The initial drop
    tau_guess = np.max(x_data) / 3.0     # Safe initial guess for the time constant
    
    params = model.make_params(
        amplitude=amp_guess,
        tau=tau_guess,
        offset=offset_guess
    )
    params['tau'].min = 0.0 # Time constant must be positive
    
    result = model.fit(y_data, params=params, x=x_data)
    tau_val = result.params['tau'].value
    
    return result, tau_val


def analyze_dispersive_shift(freqs: np.ndarray, s21_without: np.ndarray, s21_with: np.ndarray):
    """Calculates the optimal readout frequency by finding the maximum dispersive shift."""
    mag_without = np.abs(s21_without)
    mag_with = np.abs(s21_with)

    # Calculate difference
    diff = mag_with - mag_without
    abs_diff = np.abs(diff)

    # Slicing [10:-10] to avoid edge artifacts when finding the peak
    idx_max_shift = np.argmax(abs_diff[10:-10]) + 10
    f_readout_hz = freqs[idx_max_shift]
    max_shift_mag = diff[idx_max_shift]

    return {
        'f_readout_hz': f_readout_hz,
        'max_shift_mag': max_shift_mag,
        'mag_without': mag_without,
        'mag_with': mag_with,
        'diff': diff,
        'abs_diff': abs_diff,
        'idx_max_shift': idx_max_shift,
        'freqs': freqs
    }

def fit_double_gaussian(data: np.ndarray, main_mu_guess: float, leak_mu_guess: float, std_guess: float, bins: int = 75):
    """Fits a 1D dataset to a sum of two Gaussians (Main state + Leakage state)."""
    counts, bin_edges = np.histogram(data, bins=bins)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = x[1] - x[0]
    
    area = np.sum(counts) * bin_width
    density = counts / area
    
    g_main = lmfit.models.GaussianModel(prefix='main_')
    g_leak = lmfit.models.GaussianModel(prefix='leak_')
    model = g_main + g_leak
    
    params = model.make_params()
    params['main_center'].set(value=main_mu_guess)
    params['main_sigma'].set(value=std_guess, min=0)
    params['main_amplitude'].set(value=0.95, min=0) 
    
    params['leak_center'].set(value=leak_mu_guess)
    params['leak_sigma'].set(value=std_guess, min=0)
    params['leak_amplitude'].set(value=0.05, min=0) 
    
    result = model.fit(density, params=params, x=x)
    return result, area

def analyze_ssro(state0_complex: np.ndarray, state1_complex: np.ndarray):
    """Analyzes SSRO data using PCA rotation, empirical thresholding, and Double Gaussian fits."""
    mean0 = np.mean(state0_complex)
    mean1 = np.mean(state1_complex)
    
    rotation = np.mod(-np.angle(mean1 - mean0), 2 * np.pi)
    threshold = (np.exp(1j * rotation) * (mean1 + mean0)).real / 2
    
    rot0 = state0_complex * np.exp(1j * rotation)
    rot1 = state1_complex * np.exp(1j * rotation)
    I_0 = rot0.real
    I_1 = rot1.real
    
    mu0_guess = np.median(I_0)
    mu1_guess = np.median(I_1)
    std_guess = np.median(np.abs(I_0 - mu0_guess)) * 1.4826
    
    fit0, area0 = fit_double_gaussian(I_0, main_mu_guess=mu0_guess, leak_mu_guess=mu1_guess, std_guess=std_guess)
    fit1, area1 = fit_double_gaussian(I_1, main_mu_guess=mu1_guess, leak_mu_guess=mu0_guess, std_guess=std_guess)
    
    p_err_0 = np.sum(I_0 > threshold) / len(I_0)
    p_err_1 = np.sum(I_1 < threshold) / len(I_1)
    assignment_fidelity = 1 - (p_err_0 + p_err_1) / 2
    
    return {
        'rotation': rotation, 'threshold': threshold,
        'fit0': fit0, 'area0': area0,
        'fit1': fit1, 'area1': area1,
        'fidelity': assignment_fidelity,
        'p_err_0': p_err_0, 'p_err_1': p_err_1,
        'rot0': rot0, 'rot1': rot1,
        'raw_mean0': mean0, 'raw_mean1': mean1
    }