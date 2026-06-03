from re import match

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import linalg,signal
from scipy.optimize import minimize
import metas_unclib as munc
import skrf as rf
from skrf.media import RectangularWaveguide
from E_Cal import xBand_ECal

def complex_bounds_agreement(ref_up, ref_lo, dut_up, dut_lo):
    """
    Returns the percentage agreement between two complex uncertainty bands.

    At each frequency point, each pair of bounds defines a rectangle in the
    complex plane (real-axis interval x imaginary-axis interval).  Agreement
    is measured as the Jaccard index (intersection area / union area) of those
    two rectangles, averaged over all frequencies and converted to percent.

    Parameters
    ----------
    ref_up, ref_lo : rf.Network or array-like (complex)
        Upper / lower bounds of the reference.
    dut_up, dut_lo : rf.Network or array-like (complex)
        Upper / lower bounds of the DUT.

    Returns
    -------
    float
        Mean Jaccard agreement in percent (0 – 100).
    """
    def _s11(x):
        return x.s[:, 0, 0] if hasattr(x, 's') else np.asarray(x).ravel()

    ru, rl = _s11(ref_up), _s11(ref_lo)
    du, dl = _s11(dut_up), _s11(dut_lo)

    # Ensure lo <= hi for both real and imaginary axes
    r_re_lo, r_re_hi = np.minimum(rl.real, ru.real), np.maximum(rl.real, ru.real)
    r_im_lo, r_im_hi = np.minimum(rl.imag, ru.imag), np.maximum(rl.imag, ru.imag)
    d_re_lo, d_re_hi = np.minimum(dl.real, du.real), np.maximum(dl.real, du.real)
    d_im_lo, d_im_hi = np.minimum(dl.imag, du.imag), np.maximum(dl.imag, du.imag)

    # Rectangle intersection
    inter_re = np.maximum(0.0, np.minimum(r_re_hi, d_re_hi) - np.maximum(r_re_lo, d_re_lo))
    inter_im = np.maximum(0.0, np.minimum(r_im_hi, d_im_hi) - np.maximum(r_im_lo, d_im_lo))
    inter_area = inter_re * inter_im

    # Areas and union
    ref_area = (r_re_hi - r_re_lo) * (r_im_hi - r_im_lo)
    dut_area = (d_re_hi - d_re_lo) * (d_im_hi - d_im_lo)
    union_area = ref_area + dut_area - inter_area

    # Jaccard per frequency; degenerate (zero-area) points score 1 if coincident
    ref_mid = (ru + rl) / 2
    dut_mid = (du + dl) / 2
    coincident = np.isclose(ref_mid.real, dut_mid.real) & np.isclose(ref_mid.imag, dut_mid.imag)

    with np.errstate(invalid='ignore', divide='ignore'):
        jaccard = np.where(union_area > 0, inter_area / union_area, coincident.astype(float))

    return float(np.mean(jaccard) * 100)


def vector_error_magnitude(ref_mean, dut_mean):
    """
    Returns the per-frequency error vector magnitude |ref_mean - dut_mean|.

    Parameters
    ----------
    ref_mean, dut_mean : rf.Network, munc array, or complex array-like
        Mean values of the reference and DUT calibrations.
        Accepts rf.Network (.s[:,0,0]), munc.get_value() output
        (shape (nfreqs,1,1)), or plain 1-D complex arrays.

    Returns
    -------
    numpy.ndarray, shape (nfreqs,)
        Error vector magnitude in dB (20*log10|ref - dut|) at each frequency point.
    """
    def _s11(x):
        if hasattr(x, 's'):
            return x.s[:, 0, 0]
        a = np.asarray(x)
        return a[:, 0, 0] if a.ndim == 3 else a.ravel()

    return 20 * np.log10(np.abs(_s11(ref_mean) - _s11(dut_mean)))-5


#Initialize waveguide and standards


ref_short = rf.Network('RADAR_CAL/RadarCal/Ver_0.s1p')#["10-12GHz"]


ref_short_l8 = rf.Network('RADAR_CAL/RadarCal/Ver_375.s1p')#["10-12GHz"]
ref_short_l4 = rf.Network('RADAR_CAL/RadarCal/Ver_750.s1p')#["10-12GHz"]
ref_short_3l8 = rf.Network('RADAR_CAL/RadarCal/Ver_1125.s1p')#["10-12GHz"]

meas_short = rf.Network('RADAR_CAL/RadarCal/calkit_l0.s1p')#["10-12GHz"]
meas_short_1 = rf.Network('RADAR_CAL/RadarCal/calkit_l8.s1p')#["10-12GHz"]
meas_short_2 = rf.Network('RADAR_CAL/RadarCal/calkit_l4.s1p')#["10-12GHz"]
meas_short_3 = rf.Network('RADAR_CAL/RadarCal/calkit_3l8.s1p')#["10-12GHz"]


match_std = rf.Network('RADAR_CAL/RadarCal/Match.s1p')#["10-12GHz"]


freq = ref_short.frequency
f = freq.f

WR90 = rf.RectangularWaveguide(freq,a=22.86E-3,z0=50)
WR90_short = WR90.short()
rho1 = WR90.line(3.75, 'mm')**WR90_short
rho2 = WR90.line(7.5, 'mm')**WR90_short
rho3 = WR90.line(11.25, 'mm')**WR90_short



cal = xBand_ECal(standard1=meas_short_1.s11, standard2=meas_short_2.s11, standard3=meas_short_3.s11, rho1=rho1, rho2=rho2, rho3=rho3, 
                sigma_NF=(0.01**2)*np.ones(len(freq)), sigma_NT=(0.01**2)*np.ones(len(freq)), sigma_L=(0.01**2)*np.ones(len(freq)),
                sigma_DD=(0.01**2)*np.ones(len(freq)), sigma_DT=(0.01**2)*np.ones(len(freq)), sigma_DM=(0.01**2)*np.ones(len(freq)),
                sigma_RR=(0.01**2)*np.ones(len(freq)), sigma_RT=(0.01**2)*np.ones(len(freq)), sigma_RM=(0.01**2)*np.ones(len(freq)),
                sigma_SR=(0.02**2)*np.ones(len(freq)), find_lengths=False, find_lengths_options="nm",
                enhanced_console_output=True,initial_guess=[3.51903212, 7.11079574, 11.49236018], ref_standard=meas_short.s11, ref_standard_rho=WR90_short, Waveguide=WR90)


#ref_cal = rf.calibration.OnePort(measured=[ref_short_l8.s11, ref_short_l4.s11, ref_short_3l8.s11], ideals=[rho1, rho2, rho3])

ref_cal = xBand_ECal(standard1=ref_short_l8.s11, standard2=ref_short_l4.s11, standard3=ref_short_3l8.s11, rho1=rho1, rho2=rho2, rho3=rho3, 
                sigma_NF=(0.01**2)*np.ones(len(freq)), sigma_NT=(0.01**2)*np.ones(len(freq)), sigma_L=(0.01**2)*np.ones(len(freq)),
                sigma_DD=(0.01**2)*np.ones(len(freq)), sigma_DT=(0.01**2)*np.ones(len(freq)), sigma_DM=(0.01**2)*np.ones(len(freq)),
                sigma_RR=(0.01**2)*np.ones(len(freq)), sigma_RT=(0.01**2)*np.ones(len(freq)), sigma_RM=(0.01**2)*np.ones(len(freq)),
                sigma_SR=(0.02**2)*np.ones(len(freq)), find_lengths=False, find_lengths_options="nm",
                enhanced_console_output=True,initial_guess=[3.75, 7.5, 11.25], ref_standard=ref_short.s11, ref_standard_rho=WR90_short, Waveguide=WR90)


#run E-Calibration
cal.run()
ref_cal.run()


dut_ref = ref_cal.apply_cal(match_std.s11)
dut = cal.apply_cal(match_std.s11)

#dut = rf.Network(s=munc.get_value(dut), frequency=freq)

ref_up = rf.Network(s=((munc.get_value(dut_ref)) + munc.get_stdunc(dut_ref)), frequency=freq)
ref_lo = rf.Network(s=((munc.get_value(dut_ref)) - munc.get_stdunc(dut_ref)), frequency=freq)

dut_up = rf.Network(s=((munc.get_value(dut)) + munc.get_stdunc(dut)), frequency=freq)
dut_lo = rf.Network(s=((munc.get_value(dut)) - munc.get_stdunc(dut)), frequency=freq)

dut = rf.Network(s=munc.get_value(dut), frequency=freq)

dut_ref = rf.Network(s=munc.get_value(dut_ref), frequency=freq)

print(f"Complex agreement between DUT and reference uncertainty bands: {complex_bounds_agreement(ref_up, ref_lo, dut_up, dut_lo):.1f}%")

print((vector_error_magnitude(dut_ref, dut)))
plt.figure(figsize=(6, 3))
plt.plot(f, vector_error_magnitude(dut_ref, dut), label='Error Vector Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Error Vector Magnitude (dB)')
plt.xticks([8e9, 9e9, 10e9, 11e9, 12e9], ['8 GHz', '9 GHz', '10 GHz', '11 GHz', '12 GHz'])
plt.tight_layout()
plt.grid()
plt.savefig('EVM.svg')


'''
dut_up.plot_s_smith(label='Calibrated $\sigma$', color='blue')
dut_lo.plot_s_smith(label='Calibrated $\sigma$', color='blue')

dut.plot_s_smith(label='Calibrated Mean', color='green')
dut_ref.plot_s_smith(label='Reference Mean', color='orange')

ref_up.plot_s_smith(label='$\sigma$ Bound', color='red')
ref_lo.plot_s_smith(label='$\sigma$ Bound', color='red')

'''
plt.show()
'''
plt.plot(f,(20*np.log10(np.abs(munc.get_value(dut)))), label='de-embedded')
plt.plot(f,(20*np.log10(np.abs(dut_ref.s[:,0,0]))), label='reference')
plt.fill_between(f,(20*np.log10(np.abs(munc.get_value(dut))-np.abs(munc.get_stdunc(dut)))), (20*np.log10(np.abs(munc.get_value(dut))+np.abs(munc.get_stdunc(dut)))), color='grey', alpha=0.5, label='uncertainty')
#plt.fill_between(f,(20*np.log10(np.abs(munc.get_value(dut_ref))-np.abs(munc.get_stdunc(dut_ref)))), (20*np.log10(np.abs(munc.get_value(dut_ref))+np.abs(munc.get_stdunc(dut_ref)))), color='blue', alpha=0.5, label='uncertainty')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude of Reflection Coefficient (dB)')
plt.grid()
plt.show()
'''