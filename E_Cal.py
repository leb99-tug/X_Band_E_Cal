from unittest import result

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import linalg,signal
from scipy.optimize import minimize
import metas_unclib as munc
import skrf as rf
from skrf.media import RectangularWaveguide
from scipy.optimize import minimize, differential_evolution
from tqdm.auto import tqdm

def s_to_t(s):
    """Convert 2-port S-parameters to T-parameters."""
    s11, s12, s21, s22 = s[..., 0, 0], s[..., 0, 1], s[..., 1, 0], s[..., 1, 1]

    t = s.copy()
    # T-parameter mapping
    t[..., 0, 0] = -(s11 * s22 - s12 * s21) / s21
    t[..., 0, 1] = s11 / s21
    t[..., 1, 0] = -s22 / s21
    t[..., 1, 1] = 1 / s21
    return t

def t_to_s(t):
    """Convert 2-port T-parameters to S-parameters."""
    t11, t12, t21, t22 = t[..., 0, 0], t[..., 0, 1], t[..., 1, 0], t[..., 1, 1]

    s = t.copy()
    # S-parameter mapping
    s[..., 0, 0] = t12 / t22
    s[..., 0, 1] = (t11 * t22 - t12 * t21) / t22
    s[..., 1, 0] = 1 / t22
    s[..., 1, 1] = -t21 / t22
    return s


def get_noise_lin_unc_matrix(sigma_NF,sigma_NT,sigma_L,freq):
    """
    Generate noise linear uncertainty matrices.

    Args:
        sigma_NF (np.ndarray): Noise figure uncertainties.
        sigma_NT (np.ndarray): Noise temperature uncertainties.
        sigma_L (np.ndarray): Loss uncertainties.
        freq (np.ndarray): Frequency values.

    Returns:
        list: List of noise linear uncertainty matrices.
    """
    nl = []
    for inx in range(len(freq)):
        # 2x2 uncertainty network following the S-param error-network convention:
        # (0,0) = additive noise floor (value=0, pure uncertainty)
        # (0,1) = multiplicative noise+loss (value=1, gain-like term)
        # (1,0) = ideal pass-through (no uncertainty)
        # (1,1) = ideal zero (no uncertainty — noise has no mismatch coupling)
        nl_00 = munc.ucomplex(value=0, covariance=[[sigma_NF[inx],0],[0,sigma_NF[inx]]])
        nl_01 = munc.ucomplex(value=1, covariance=[[sigma_NT[inx]+sigma_L[inx],0],[0,sigma_NT[inx]+sigma_L[inx]]])
        nl_10 = munc.ucomplex(value=1, covariance=[[0,0],[0,0]])
        nl_11 = munc.ucomplex(value=0, covariance=[[0,0],[0,0]])
        _nl = np.array([[nl_00, nl_01],[nl_10, nl_11]])
        nl.append(_nl)
    return nl

def get_drift_unc_matrix(sigma_DD,sigma_DT,sigma_DM,freq):
    """
    Generate drift uncertainty matrices.

    Args:
        sigma_DD (np.ndarray): Drift uncertainties for directivity.
        sigma_DT (np.ndarray): Drift uncertainties for tracking.
        sigma_DM (np.ndarray): Drift uncertainties for mismatch.
        freq (np.ndarray): Frequency values.

    Returns:
        list: List of drift uncertainty matrices.
    """
    dd=[]
    for inx in range(len(freq)):
        # Same 2x2 structure as noise, but (1,1) is non-zero:
        # (0,0)=directivity drift (additive), (0,1)=tracking drift (multiplicative),
        # (1,1)=mismatch drift — VNA source-match drifts introduce a return-path coupling term
        dd_00 = munc.ucomplex(value=0, covariance=[[sigma_DD[inx],0],[0,sigma_DD[inx]]])
        dd_01 = munc.ucomplex(value=1, covariance=[[sigma_DT[inx],0],[0,sigma_DT[inx]]])
        dd_10 = munc.ucomplex(value=1, covariance=[[0,0],[0,0]])
        dd_11 = munc.ucomplex(value=0, covariance=[[sigma_DM[inx],0],[0,sigma_DM[inx]]])
        _dd = np.array([[dd_00, dd_01],[dd_10, dd_11]])
        dd.append(_dd)
    return dd

def get_rep_unc_matrix(sigma_RR,sigma_RT,sigma_RM,freq):
    """
    Generate repeatability uncertainty matrices.

    Args:
        sigma_RR (np.ndarray): Repeatability uncertainties for reflection.
        sigma_RT (np.ndarray): Repeatability uncertainties for tracking.
        sigma_RM (np.ndarray): Repeatability uncertainties for mismatch.
        freq (np.ndarray): Frequency values.

    Returns:
        list: List of repeatability uncertainty matrices.
    """
    rep = []
    for inx in range(len(freq)):
        # Both off-diagonal terms carry sigma_RT: repeatability affects forward and
        # reverse transmission paths equally (connector re-seating is non-directional)
        rep_00 = munc.ucomplex(value=0, covariance=[[sigma_RR[inx],0],[0,sigma_RR[inx]]])
        rep_01 = munc.ucomplex(value=1, covariance=[[sigma_RT[inx],0],[0,sigma_RT[inx]]])
        rep_10 = munc.ucomplex(value=1, covariance=[[sigma_RT[inx],0],[0,sigma_RT[inx]]])
        rep_11 = munc.ucomplex(value=0, covariance=[[sigma_RM[inx],0],[0,sigma_RM[inx]]])
        _rep = np.array([[rep_00, rep_01],[rep_10, rep_11]])
        rep.append(_rep)
    return rep

def get_standard_unc_matrix(sigma_SR,freq):
    """
    Generate standard uncertainty matrices.

    Args:
        sigma_SR (np.ndarray): Standard uncertainties for reflection.
        freq (np.ndarray): Frequency values.

    Returns:
        list: List of standard uncertainty matrices.
    """
    sr = []
    for inx in range(len(freq)):
        # Only (0,0) carries uncertainty: the standard's imperfection is purely additive
        # (offset in reflection coefficient), not a tracking or mismatch error
        sr_00 = munc.ucomplex(value=0, covariance=[[sigma_SR[inx],0],[0,sigma_SR[inx]]])
        sr_01 = munc.ucomplex(value=1, covariance=[[0,0],[0,0]])
        sr_10 = munc.ucomplex(value=1, covariance=[[0,0],[0,0]])
        sr_11 = munc.ucomplex(value=0, covariance=[[0,0],[0,0]])
        _sr = np.array([[sr_00, sr_01],[sr_10, sr_11]])
        sr.append(_sr)
    return sr

def de_emb(ED, EM, ET, cov_output, DUT: rf.Network):
    """
    Perform de-embedding calculations.

    Args:
        ED (np.ndarray): Calibration parameter ED.
        EM (np.ndarray): Calibration parameter EM.
        ET (np.ndarray): Calibration parameter ET.
        cov_output (list): Covariance matrices for the output.
        DUT (rf.Network): Device Under Test network data.

    Returns:
        np.ndarray: De-embedded reflection coefficients.
    """
    freq_vals = DUT.frequency.f  # Ensure freq_vals is defined before usage
    unc_matrix = np.zeros((len(freq_vals), 2, 2), dtype=object)
    unc_matrix = cov_output
    M = np.array([x.s.squeeze() for x in DUT])
    rho = []
    for inx in range(len(freq_vals)):
        # 1-port error-model correction: Γ = (M - ED) / (ET + EM*(M - ED))
        _rho = (M[inx] - ED[inx]) / (ET[inx] + EM[inx] * (M[inx] - ED[inx]))
        # Propagate residual output uncertainty through the error network via bilinear transform:
        # Γ_out = M00 + M01*M10*Γ / (1 - M11*Γ) — this is the standard 2-port signal-flow reduction
        _rho_dut = unc_matrix[inx, 0, 0] + (unc_matrix[inx, 0, 1] * unc_matrix[inx, 1, 0] * _rho) / (1 - unc_matrix[inx, 1, 1] * _rho)
        rho.append(_rho_dut)
    rho = np.array(rho)
    return rho


def SSS(Gamma1: rf.Network, Gamma2: rf.Network, Gamma3: rf.Network,
        rho1: rf.Network, rho2: rf.Network, rho3: rf.Network):
    """
    Compute calibration parameters (ED, ET, EM) based on measured and ideal reflection coefficients.

    Args:
        Gamma1 (rf.Network): Measured reflection coefficient for the first standard.
        Gamma2 (rf.Network): Measured reflection coefficient for the second standard.
        Gamma3 (rf.Network): Measured reflection coefficient for the third standard.
        rho1 (rf.Network): Ideal reflection coefficient for the first standard.
        rho2 (rf.Network): Ideal reflection coefficient for the second standard.
        rho3 (rf.Network): Ideal reflection coefficient for the third standard.

    Returns:
        tuple: A tuple containing:
            - ED (np.ndarray): Calibration parameter ED.
            - ET (np.ndarray): Calibration parameter ET.
            - EM (np.ndarray): Calibration parameter EM.

    Raises:
        ZeroDivisionError: If a division by zero occurs during the computation.
    """
    # Placeholder for uncertainties (must be replaced by actual calculation)
    ED = []
    ET= []
    EM = []

    f  = Gamma1.frequency.f
    G1 = np.array([x.s.squeeze() for x in Gamma1])
    G2 = np.array([x.s.squeeze() for x in Gamma2])
    G3 = np.array([x.s.squeeze() for x in Gamma3])


    rho1 = np.array([x.s.squeeze() for x in rho1])
    rho2 = np.array([x.s.squeeze() for x in rho2])
    rho3 = np.array([x.s.squeeze() for x in rho3])



    _EM = np.zeros(len(f), dtype=complex)
    _ET = np.zeros(len(f), dtype=complex)
    _ED = np.zeros(len(f), dtype=complex)
    for inx, f in enumerate(f):
        r1, r2, r3 = rho1[inx], rho2[inx], rho3[inx]
        g1, g2, g3 = G1[inx], G2[inx], G3[inx]

        # Step 1: auxiliary variable K = [(G1-G2)/(G2-G3)] * [(r2-r3)/(r1-r2)]
        # K encodes how the measured spacing between standards compares to the ideal spacing
        k_numerator = (g1 - g2) * (r2 - r3)
        k_denominator = (g2 - g3) * (r1 - r2)

        if k_denominator == 0:
            raise ZeroDivisionError("Division by zero in K calculation (check r differences).")

        K = k_numerator / k_denominator

        # Step 2: source match EM
        em_denominator = (r3 - K * r1)
        if em_denominator == 0:
            raise ZeroDivisionError("Singularity in EM: denominator is zero.")

        _EM[inx] = (1 - K) / em_denominator

        # Step 3: tracking ET = (G1-G2) * (1-EM*r1) * (1-EM*r2) / (r1-r2)
        _ET[inx] = (g1 - g2) * (1 - _EM[inx] * r1) * (1 - _EM[inx] * r2) / (r1 - r2)

        # Step 4: directivity ED = G1 - ET*r1 / (1 - EM*r1)
        _ED[inx] = g1 - (_ET[inx] * r1) / (1 - _EM[inx] * r1)

        ED.append(_ED[inx])
        ET.append(_ET[inx])
        EM.append(_EM[inx])

    ED = np.array(ED)
    ET = np.array(ET)
    EM = np.array(EM)



    return ED, ET, EM

def unc_SSS(Gamma1, Gamma2, Gamma3,
            rho1, rho2, rho3, freq,
            unc_Gamma1=None, unc_Gamma2=None, unc_Gamma3=None,
            unc_rho1=None, unc_rho2=None, unc_rho3=None):
    """
    Compute calibration parameters (ED, ET, EM) with uncertainty propagation.

    This function calculates the calibration parameters ED, ET, and EM based on the measured and ideal reflection coefficients,
    while accounting for uncertainties in both the measured and ideal coefficients.

    Args:
        Gamma1 (rf.Network): Measured reflection coefficient for the first standard.
        Gamma2 (rf.Network): Measured reflection coefficient for the second standard.
        Gamma3 (rf.Network): Measured reflection coefficient for the third standard.
        rho1 (rf.Network): Ideal reflection coefficient for the first standard.
        rho2 (rf.Network): Ideal reflection coefficient for the second standard.
        rho3 (rf.Network): Ideal reflection coefficient for the third standard.
        freq (np.ndarray): Frequency values.
        unc_Gamma1 (np.ndarray, optional): Uncertainty covariance matrix for Gamma1. Defaults to None.
        unc_Gamma2 (np.ndarray, optional): Uncertainty covariance matrix for Gamma2. Defaults to None.
        unc_Gamma3 (np.ndarray, optional): Uncertainty covariance matrix for Gamma3. Defaults to None.
        unc_rho1 (np.ndarray, optional): Uncertainty covariance matrix for rho1. Defaults to None.
        unc_rho2 (np.ndarray, optional): Uncertainty covariance matrix for rho2. Defaults to None.
        unc_rho3 (np.ndarray, optional): Uncertainty covariance matrix for rho3. Defaults to None.

    Raises:
        ZeroDivisionError: If a division by zero occurs during the computation of K, EM, or ET.

    Returns:
        tuple: A tuple containing:
            - ED (np.ndarray): Calibration parameter ED with uncertainties.
            - ET (np.ndarray): Calibration parameter ET with uncertainties.
            - EM (np.ndarray): Calibration parameter EM with uncertainties.
    """

    nf = len(freq)



    u_G1 = [None] * nf
    u_G2 = [None] * nf
    u_G3 = [None] * nf
    u_rho1 = [None] * nf
    u_rho2 = [None] * nf
    u_rho3 = [None] * nf

    if unc_Gamma1 is not None:
        # Wrap plain complex values into munc.ucomplex so the SSS algebra
        # automatically propagates covariance through every arithmetic operation
        for inx in range(nf):
            u_G1[inx] = munc.ucomplex(value=Gamma1[inx], covariance=unc_Gamma1[inx,:,:])
            u_G2[inx] = munc.ucomplex(value=Gamma2[inx], covariance=unc_Gamma2[inx,:,:])
            u_G3[inx] = munc.ucomplex(value=Gamma3[inx], covariance=unc_Gamma3[inx,:,:])
    else:
        u_G1 = Gamma1
        u_G2 = Gamma2
        u_G3 = Gamma3

    if unc_rho1 is not None:
        for inx in range(nf):
            u_rho1[inx] = munc.ucomplex(value=rho1[inx], covariance=unc_rho1[inx,:,:])
            u_rho2[inx] = munc.ucomplex(value=rho2[inx], covariance=unc_rho2[inx,:,:])
            u_rho3[inx] = munc.ucomplex(value=rho3[inx], covariance=unc_rho3[inx,:,:])
    else:
        u_rho1 = rho1
        u_rho2 = rho2
        u_rho3 = rho3

    ED = []
    ET = []
    EM = []

    # Same SSS algebra as in SSS(), but operating on munc.ucomplex objects
    # so uncertainty propagates through K → EM → ET → ED automatically
    for inx in range(nf):
        r1, r2, r3 = u_rho1[inx], u_rho2[inx], u_rho3[inx]
        g1, g2, g3 = u_G1[inx], u_G2[inx], u_G3[inx]

        k_numerator = (g1 - g2) * (r2 - r3)
        k_denominator = (g2 - g3) * (r1 - r2)

        if k_denominator == 0:
            raise ZeroDivisionError("Division by zero in K calculation.")

        K = k_numerator / k_denominator

        em_denominator = (r3 - K * r1)
        if em_denominator == 0:
            raise ZeroDivisionError("Singularity in EM: denominator is zero.")

        _EM = (1 - K) / em_denominator
        _ET = (g1 - g2) * (1 - _EM * r1) * (1 - _EM * r2) / (r1 - r2)
        _ED = g1 - (_ET * r1) / (1 - _EM * r1)

        ED.append(_ED)
        ET.append(_ET)
        EM.append(_EM)

    return np.array(ED), np.array(ET), np.array(EM)


def find_calibration_lengths(
    standard1: rf.Network,
    standard2: rf.Network,
    standard3: rf.Network,
    initial_guess: list,
    ref_standard: rf.Network,
    ref_standard_rho: rf.Network,
    Waveguide: rf.media,
    method: str = "de+nm",
    bounds: list | None = None,
    show_progress: bool = True,
    enhanced_console_output: bool = False,
    de_kwargs: dict | None = None,
    nm_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Find calibration lengths based on a reference standard.

    Args:
        standard1, standard2, standard3: Measured offset-short networks.
        initial_guess: Initial guess for lengths [mm].
        ref_standard: Measured network data for the reference standard.
        ref_standard_rho: Ideal reflection network for the reference standard.
        Waveguide: Waveguide media for length calculations.
        method: Optimization strategy. One of:
            - "nm"     : Nelder-Mead only, starts from initial_guess (fast, local)
            - "de"     : Differential evolution only (global, no polish)
            - "de+nm"  : DE for global search, then Nelder-Mead polish (default,
                         most robust)
        bounds: List of (low, high) tuples per length [mm]. Required for "de"
            and "de+nm". If None, defaults to ±30% around each initial guess.
        show_progress: Show tqdm progress bar.
        enhanced_console_output: Print extra info per iteration.
        de_kwargs: Overrides for differential_evolution.
        nm_kwargs: Overrides for Nelder-Mead options.

    Returns:
        np.ndarray: Optimized calibration lengths [mm].
    """
    if method not in {"nm", "de", "de+nm"}:
        raise ValueError(
            f"method must be 'nm', 'de', or 'de+nm', got {method!r}"
        )
    DEG_PER_RAD = 180 / np.pi
    measured = [standard1.s11, standard2.s11, standard3.s11]
    ref_meas = ref_standard.s11

    def goal_function(l):
        rho1 = Waveguide.line(l[0], 'mm') ** ref_standard_rho
        rho2 = Waveguide.line(l[1], 'mm') ** ref_standard_rho
        rho3 = Waveguide.line(l[2], 'mm') ** ref_standard_rho

        cal = rf.calibration.OnePort(
            measured=measured,
            ideals=[rho1, rho2, rho3],
        )
        cal.run()
        DUT = cal.apply_cal(ref_meas)

        # Cost is sum-of-squared phase errors (not magnitude): the line length
        # information is entirely encoded in phase; magnitude stays ~1 regardless of length
        angle_dut = np.unwrap(np.angle(DUT.s.squeeze(),deg=True),discont=-180)
        angle_ref = np.unwrap(np.angle(ref_standard_rho.s.squeeze(),deg=True),discont=-180)
        err = np.sum((angle_ref-angle_dut)**2)

        if enhanced_console_output:
            print(f"l = {np.array(l)}, error = {err:.6g}")
        return err

    # Progress bar wrapper that tracks the running best value
    class _Progress:
        def __init__(self, total, desc):
            self.bar = tqdm(total=total, desc=desc, unit="eval",
                            disable=not show_progress)
            self.best = np.inf

        def wrap(self, f):
            def inner(x):
                v = f(x)
                if v < self.best:
                    self.best = v
                self.bar.set_postfix(best=f"{self.best:.4g}",
                                     current=f"{v:.4g}")
                self.bar.update(1)
                return v
            return inner

        def close(self):
            self.bar.close()

    n = len(initial_guess)

    # Default bounds: ±30% around each initial guess; avoids negative lengths
    if bounds is None and method in {"de", "de+nm"}:
        bounds = [
            (g * 0.7, g * 1.3) if g > 0 else (0, 20)
            for g in initial_guess
        ]

    # --- Global phase: Differential Evolution (avoids local minima) ---
    if method in {"de", "de+nm"}:
        de_defaults = dict(
            popsize=20,
            maxiter=150,
            tol=1e-8,
            seed=42,
            polish=False,  # polish=False: let Nelder-Mead do the fine tuning
            workers=1,
            disp=False,
        )
        if de_kwargs:
            de_defaults.update(de_kwargs)

        de_total = de_defaults['popsize'] * n * (de_defaults['maxiter'] + 1)
        prog = _Progress(total=de_total, desc="Differential Evolution")
        try:
            de_result = differential_evolution(
                prog.wrap(goal_function),
                bounds=bounds,
                **de_defaults,
            )
        finally:
            prog.close()

        x0 = de_result.x
        if enhanced_console_output:
            print(f"DE result: x={x0}, f={de_result.fun:.6g}")

        if method == "de":
            return x0
    else:
        x0 = np.asarray(initial_guess, dtype=float)

    # --- Local phase: Nelder-Mead polish starting from DE result ---
    nm_defaults = dict(xatol=1e-9, fatol=1e-9, maxiter=2000)
    if nm_kwargs:
        nm_defaults.update(nm_kwargs)

    prog = _Progress(total=nm_defaults['maxiter'], desc="Nelder-Mead")
    try:
        result = minimize(
            prog.wrap(goal_function),
            x0=x0,
            method='Nelder-Mead',
            options=nm_defaults,
            bounds=bounds,
        )
    finally:
        prog.close()

    if enhanced_console_output:
        print(f"Final: x={result.x}, f={result.fun:.6g}")

        rho1 = Waveguide.line(result.x[0], 'mm') ** ref_standard_rho
        rho2 = Waveguide.line(result.x[1], 'mm') ** ref_standard_rho
        rho3 = Waveguide.line(result.x[2], 'mm') ** ref_standard_rho

        cal = rf.calibration.OnePort(
            measured=measured,
            ideals=[rho1, rho2, rho3],
        )
        cal.run()
        DUT = cal.apply_cal(ref_meas)

        plt.figure(figsize=(10, 6))
        plt.title("Phase Comparison: Reference Standard vs. Optimized Calibrated DUT")
        ref_standard_rho.plot_s_deg_unwrap(label="Reference Standard Rho")
        DUT.plot_s_deg_unwrap(label="Calibrated DUT")

        plt.show()

    return result.x

class xBand_ECal:
    """
    Class for X-Band calibration and uncertainty calculations.

    This class handles the calibration standards, their uncertainties, and the frequency data for X-Band calibration.
    """
    def __init__(self, standard1: rf.Network, standard2: rf.Network, standard3: rf.Network,
                rho1: rf.Network, rho2: rf.Network, rho3: rf.Network,
                sigma_NF=None, sigma_NT=None, sigma_L=None,
                sigma_DD=None, sigma_DT=None, sigma_DM=None,
                sigma_RR=None, sigma_RT=None, sigma_RM=None,
                sigma_SR=None, find_lengths=False, initial_guess=None,
                ref_standard:rf.Network=None, ref_standard_rho:rf.Network=None, Waveguide:rf.media = None,
                enhanced_console_output:bool=False, find_lengths_options:dict=None):
        """
        Initialize the xBand_ECal class with calibration standards, ideal reflection coefficients, and uncertainties.

        Args:
            standard1 (rf.Network): Measured network data for the first calibration standard.
            standard2 (rf.Network): Measured network data for the second calibration standard.
            standard3 (rf.Network): Measured network data for the third calibration standard.
            rho1 (rf.Network): Ideal reflection coefficient for the first calibration standard.
            rho2 (rf.Network): Ideal reflection coefficient for the second calibration standard.
            rho3 (rf.Network): Ideal reflection coefficient for the third calibration standard.
            sigma_NF (np.ndarray, optional): Noise floor uncertainties. Defaults to None.
            sigma_NT (np.ndarray, optional): Noise tracking uncertainties. Defaults to None.
            sigma_L (np.ndarray, optional): Linearity uncertainties. Defaults to None.
            sigma_DD (np.ndarray, optional): Drift uncertainties for directivity. Defaults to None.
            sigma_DT (np.ndarray, optional): Drift uncertainties for tracking. Defaults to None.
            sigma_DM (np.ndarray, optional): Drift uncertainties for mismatch. Defaults to None.
            sigma_RR (np.ndarray, optional): Repeatability uncertainties for reflection. Defaults to None.
            sigma_RT (np.ndarray, optional): Repeatability uncertainties for tracking. Defaults to None.
            sigma_RM (np.ndarray, optional): Repeatability uncertainties for mismatch. Defaults to None.
            sigma_SR (np.ndarray, optional): Standard uncertainties for reflection. Defaults to None.
            find_lengths (bool, optional): Flag to indicate whether to find lengths. Defaults to False.
            find_lengths_options (dict, optional): Options for the length optimization process. Defaults to "de+nm" others are "de", "nm".
            initial_guess (list, optional): Initial guess for the optimization of lengths. Defaults to None.
            ref_standard (rf.Network, optional): Reference standard network data. Defaults to None.
            ref_standard_rho (rf.Network, optional): Ideal reflection coefficient for the reference standard. Defaults to None.
            Waveguide (rf.media, optional): Waveguide media for length calculations. Defaults to None.
            enhanced_console_output (bool, optional): Flag to enable enhanced console output during optimization. Defaults to False.
        """

        ####################################################################################################
        #----- Calibration standards and ideal reflection coefficients ------------------------------------#
        ####################################################################################################
        self.standard1 = np.array([x.s.squeeze() for x in standard1])
        self.standard2 = np.array([x.s.squeeze() for x in standard2])
        self.standard3 = np.array([x.s.squeeze() for x in standard3])


        self.freq = standard1.frequency.f
        self.enhanced_console_output = enhanced_console_output
        self.find_lengths = find_lengths
        self.ref_standard = np.array([x.s.squeeze() for x in ref_standard])
        self.ref_standard_rho = ref_standard_rho

        ####################################################################################################
        #----- Determine actual offset-short lengths (optimisation) --------------------------------------#
        ####################################################################################################

        if self.find_lengths and self.ref_standard is not None and self.ref_standard_rho is not None and Waveguide is not None:
            if self.enhanced_console_output: print("Finding calibration lengths for SSS Cal...")
            self.lengths = find_calibration_lengths(standard1=standard1, standard2=standard2, standard3=standard3,
                                                    initial_guess=initial_guess, ref_standard=ref_standard,
                                                    ref_standard_rho=ref_standard_rho, Waveguide=Waveguide,
                                                    enhanced_console_output=self.enhanced_console_output, method=find_lengths_options)
            print(f"Optimized lengths: {self.lengths[0]:.6f} mm, {self.lengths[1]:.6f} mm, {self.lengths[2]:.6f} mm")
            self.rho1 = Waveguide.line(self.lengths[0], 'mm')**self.ref_standard_rho
            self.rho2 = Waveguide.line(self.lengths[1], 'mm')**self.ref_standard_rho
            self.rho3 = Waveguide.line(self.lengths[2], 'mm')**self.ref_standard_rho
            self.rho1 = np.array([x.s.squeeze() for x in self.rho1])
            self.rho2 = np.array([x.s.squeeze() for x in self.rho2])
            self.rho3 = np.array([x.s.squeeze() for x in self.rho3])
        else:
            self.rho1 = np.array([x.s.squeeze() for x in rho1])
            self.rho2 = np.array([x.s.squeeze() for x in rho2])
            self.rho3 = np.array([x.s.squeeze() for x in rho3])

        ####################################################################################################
        #----- Build uncertainty matrices ----------------------------------------------------------------#
        ####################################################################################################
        if sigma_NF is not None and sigma_NT is not None and sigma_L is not None:
            if self.enhanced_console_output: print("Generating noise and linearity uncertainty matrix...")
            self.noise_lin_unc_matrix = get_noise_lin_unc_matrix(sigma_NF=sigma_NF, sigma_NT=sigma_NT, sigma_L=sigma_L, freq=self.freq)
        else:
            # Zero variances: noise/linearity uncertainty disabled
            self.noise_lin_unc_matrix = get_noise_lin_unc_matrix(sigma_NF=0*np.ones(len(self.freq)), sigma_NT=0*np.ones(len(self.freq)), sigma_L=0*np.ones(len(self.freq)), freq=self.freq)

        if sigma_DD is not None and sigma_DT is not None and sigma_DM is not None:
            if self.enhanced_console_output: print("Generating drift uncertainty matrix...")
            self.drift_unc_matrix = get_drift_unc_matrix(sigma_DD=sigma_DD, sigma_DT=sigma_DT, sigma_DM=sigma_DM, freq=self.freq)
        else:
            self.drift_unc_matrix = get_drift_unc_matrix(sigma_DD=0*np.ones(len(self.freq)), sigma_DT=0*np.ones(len(self.freq)), sigma_DM=0*np.ones(len(self.freq)), freq=self.freq)

        if sigma_RR is not None and sigma_RT is not None and sigma_RM is not None:
            if self.enhanced_console_output: print("Generating repeatability uncertainty matrix...")
            self.rep_unc_matrix = get_rep_unc_matrix(sigma_RR=sigma_RR, sigma_RT=sigma_RT, sigma_RM=sigma_RM, freq=self.freq)
        else:
            self.rep_unc_matrix = get_rep_unc_matrix(sigma_RR=0*np.ones(len(self.freq)), sigma_RT=0*np.ones(len(self.freq)), sigma_RM=0*np.ones(len(self.freq)), freq=self.freq)
        if sigma_SR is not None:
            if self.enhanced_console_output: print("Generating standard uncertainty matrix...")
            self.standard_unc_matrix = get_standard_unc_matrix(sigma_SR=sigma_SR, freq=self.freq)
        else:
            self.standard_unc_matrix = get_standard_unc_matrix(sigma_SR=0*np.ones(len(self.freq)), freq=self.freq)

        ####################################################################################################
        #----- Combine output-side uncertainty networks (output_matrix) ----------------------------------#
        ####################################################################################################

        # output_matrix stores munc.ucomplex objects, so dtype=object is required
        self.output_matix = np.zeros((len(self.freq), 2,2), dtype=object)

        # Chain drift → repeatability → standard uncertainty in T-domain via matrix multiply.
        # T-domain (wave-cascade) multiplication is correct here because these three error sources
        # appear in series at the reference plane; S-domain multiply would be wrong.
        for inx in range(len(self.freq)):
            self.drift_t_matrix = s_to_t(self.drift_unc_matrix[inx])
            self.rep_t_matrix = s_to_t(self.rep_unc_matrix[inx])
            self.standard_t_matrix = s_to_t(self.standard_unc_matrix[inx])
            self.output_matix[inx] = self.drift_t_matrix @ self.rep_t_matrix @ self.standard_t_matrix

        # Convert back to S-parameters for use in the bilinear de-embedding formula
        self.output_matix = t_to_s(self.output_matix)

        self.cov_rho1  = np.zeros((len(self.freq), 2,2))
        self.cov_rho2  = np.zeros((len(self.freq), 2,2))
        self.cov_rho3  = np.zeros((len(self.freq), 2,2))

        for inx in range(len(self.freq)):
            # Apply the combined output error network to each ideal standard via the bilinear transform:
            # Γ_perturbed = M00 + M01*M10*rho / (1 - M11*rho)
            # munc.get_covariance then extracts the GUM uncertainty of the perturbed value
            rho1_meas = self.output_matix[inx,0,0]+(self.output_matix[inx,0,1]*self.output_matix[inx,1,0]*self.rho1[inx])/(1-self.output_matix[inx,1,1]*self.rho1[inx])
            self.cov_rho1[inx] = munc.get_covariance(rho1_meas)
            rho2_meas = self.output_matix[inx,0,0]+(self.output_matix[inx,0,1]*self.output_matix[inx,1,0]*self.rho2[inx])/(1-self.output_matix[inx,1,1]*self.rho2[inx])
            self.cov_rho2[inx] = munc.get_covariance(rho2_meas)
            rho3_meas = self.output_matix[inx,0,0]+(self.output_matix[inx,0,1]*self.output_matix[inx,1,0]*self.rho3[inx])/(1-self.output_matix[inx,1,1]*self.rho3[inx])
            self.cov_rho3[inx] = munc.get_covariance(rho3_meas)

        # input_matrix holds the noise/linearity uncertainty (acts on the measured standards, not the ideals)
        self.input_matix = np.zeros((len(self.freq), 2,2), dtype=object)


        self.cov_standard1  = np.zeros((len(self.freq), 2,2))
        self.cov_standard2  = np.zeros((len(self.freq), 2,2))
        self.cov_standard3  = np.zeros ((len(self.freq), 2,2))

        for inx in range(len(self.freq)):
            self.input_matix[inx] = self.noise_lin_unc_matrix[inx]

        for inx in range(len(self.freq)):
            # Back-transform: find what noise-perturbed Γ the VNA actually sees for each standard,
            # then extract covariance — this is the input-side bilinear transform (inverse direction)
            mes_standard1 = (self.standard1[inx]-self.input_matix[inx,0,0])/(self.input_matix[inx,0,1]*self.input_matix[inx,1,0]+self.input_matix[inx,1,1]*(self.standard1[inx]-self.input_matix[inx,0,0]))
            mes_standard2 = (self.standard2[inx]-self.input_matix[inx,0,0])/(self.input_matix[inx,0,1]*self.input_matix[inx,1,0]+self.input_matix[inx,1,1]*(self.standard2[inx]-self.input_matix[inx,0,0]))
            mes_standard3 = (self.standard3[inx]-self.input_matix[inx,0,0])/(self.input_matix[inx,0,1]*self.input_matix[inx,1,0]+self.input_matix[inx,1,1]*(self.standard3[inx]-self.input_matix[inx,0,0]))
            self.cov_standard1[inx] = munc.get_covariance(mes_standard1)
            self.cov_standard2[inx] = munc.get_covariance(mes_standard2)
            self.cov_standard3[inx] = munc.get_covariance(mes_standard3)

        if self.enhanced_console_output: print("Initialization complete.")

    def run(self):
        """
        Perform the calibration process.

        This method calculates the calibration parameters (ED, ET, EM) using the measured and ideal reflection coefficients,
        along with their associated uncertainties.

        The calibration parameters are computed using the `unc_SSS` function, which accounts for uncertainties in the
        measured and ideal reflection coefficients.
        """
        self.ED, self.ET, self.EM = unc_SSS(Gamma1=self.standard1, Gamma2=self.standard2, Gamma3=self.standard3,
                                        rho1=self.rho1, rho2=self.rho2, rho3=self.rho3,freq=self.freq,
                                        unc_Gamma1=self.cov_standard1, unc_Gamma2=self.cov_standard2, unc_Gamma3=self.cov_standard3,
                                        unc_rho1=self.cov_rho1, unc_rho2=self.cov_rho2, unc_rho3=self.cov_rho3)

    def apply_cal(self, DUT: rf.Network):
        """
        Apply the calibration to the Device Under Test (DUT).

        This method uses the calibration parameters (ED, EM, ET) and the covariance output matrix to de-embed the reflection coefficients of the DUT.

        Args:
            DUT (rf.Network): Measured network data for the Device Under Test.

        Returns:
            np.ndarray: De-embedded reflection coefficients of the DUT.
        """
        self.rho_deembedded = de_emb(ED=self.ED, EM=self.EM, ET=self.ET, cov_output=self.output_matix, DUT=DUT)
        return self.rho_deembedded
