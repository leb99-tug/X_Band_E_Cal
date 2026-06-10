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

    return 20 * np.log10(np.abs(_s11(ref_mean) - _s11(dut_mean)))


#Initialize waveguide and standards


ref_short = rf.Network('090626/Ver_0.s1p')["8.4-12.5GHz"]#["8.4-12GHz"]


ref_short_l8 = rf.Network('090626/Ver_375.s1p')["8.4-12.5GHz"]
ref_short_l4 = rf.Network('090626/Ver_750.s1p')["8.4-12.5GHz"]
ref_short_3l8 = rf.Network('090626/Ver_1125.s1p')["8.4-12.5GHz"]

meas_short = rf.Network('090626/calkit_l0.s1p')["8.4-12.5GHz"]
meas_short_1 = rf.Network('090626/calkit_l8.s1p')["8.4-12.5GHz"]
meas_short_2 = rf.Network('090626/calkit_l4.s1p')["8.4-12.5GHz"]
meas_short_3 = rf.Network('090626/calkit_3l8.s1p')["8.4-12.5GHz"]


match_std = rf.Network('090626/Match.s1p')["8.4-12.5GHz"]


freq = ref_short.frequency
f = freq.f

WR90 = rf.RectangularWaveguide(freq,a=22.86E-3,z0=50)
WR90_short = rf.Network(s=(-1-0.000000000001j)*np.ones(len(freq)), frequency=freq, z0=50)
rho1 = WR90.line(3.75, 'mm')**WR90_short
rho2 = WR90.line(7.5, 'mm')**WR90_short
rho3 = WR90.line(11.25, 'mm')**WR90_short

rho1_cal = WR90.line(3.75, 'mm')**WR90_short
rho2_cal = WR90.line(7.5, 'mm')**WR90_short
rho3_cal = WR90.line(11.25, 'mm')**WR90_short




cal = xBand_ECal(standard1=meas_short_1.s11, standard2=meas_short_2.s11, standard3=meas_short_3.s11, rho1=rho1_cal, rho2=rho2_cal, rho3=rho3_cal, 
                sigma_NF=(0.000001**2)*np.ones(len(freq)), sigma_NT=(0.000025**2)*np.ones(len(freq)), sigma_L=(0.0012**2)*np.ones(len(freq)),
                sigma_DD=(0.002**2)*np.ones(len(freq)), sigma_DT=(0.0125**2)*np.ones(len(freq)), sigma_DM=(0.025**2)*np.ones(len(freq)),
                sigma_RR=(0.004**2)*np.ones(len(freq)), sigma_RT=(0.0000001**2)*np.ones(len(freq)), sigma_RM=(0.004**2)*np.ones(len(freq)),
                sigma_SR=(0.05**2)*np.ones(len(freq)), find_lengths=True, find_lengths_options="nm",
                enhanced_console_output=True,initial_guess=[3.75, 7.25 , 11.25], ref_standard=meas_short.s11, ref_standard_rho=WR90_short, Waveguide=WR90)


#ref_cal = rf.calibration.OnePort(measured=[ref_short_l8.s11, ref_short_l4.s11, ref_short_3l8.s11], ideals=[rho1, rho2, rho3])

ref_cal = xBand_ECal(standard1=ref_short_l8.s11, standard2=ref_short_l4.s11, standard3=ref_short_3l8.s11, rho1=rho1, rho2=rho2, rho3=rho3, 
                sigma_NF=(0.000001**2)*np.ones(len(freq)), sigma_NT=(0.000025**2)*np.ones(len(freq)), sigma_L=(0.0012**2)*np.ones(len(freq)),
                sigma_DD=(0.002**2)*np.ones(len(freq)), sigma_DT=(0.0125**2)*np.ones(len(freq)), sigma_DM=(0.025**2)*np.ones(len(freq)),
                sigma_RR=(0.004**2)*np.ones(len(freq)), sigma_RT=(0.0000001**2)*np.ones(len(freq)), sigma_RM=(0.004**2)*np.ones(len(freq)),
                sigma_SR=(0.05**2)*np.ones(len(freq)), find_lengths=False, find_lengths_options="de",
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

plt.figure(figsize=(6, 3))
plt.plot(f, vector_error_magnitude(dut_ref, dut), label='Error Vector Magnitude (dB)')
plt.hlines(-20, f[0], f[-1], colors='red', linestyles='dashed', label='-20 dB Threshold')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Error Vector Magnitude (dB)')
plt.xticks([8.5e9, 9.5e9, 10.5e9, 11.5e9, 12.5e9], ['8.5', '9.5', '10.5', '11.5', '12.5'])
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('EVM.svg')
plt.show()

'''

#dut_up.plot_s_smith(label='Calibrated $\sigma$', color='blue')
#dut_lo.plot_s_smith(label='Calibrated $\sigma$', color='blue')

dut.plot_s_db(label='Calibrated Mean', color='green')
#dut_ref.plot_s_smith(label='Reference Mean', color='orange')

ref_up.plot_s_db(label='$\sigma$ Reference', color='red')
ref_lo.plot_s_db(label='$\sigma$ Reference', color='red')
plt.tight_layout()
plt.legend()
plt.savefig('Smith.svg')
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(f,(20*np.log10(np.abs(munc.get_value(dut)))), label='Measured (Auto-Cal-Unit)', color='blue')
#plt.plot(f,(20*np.log10(np.abs(dut_ref.s[:,0,0]))), label='reference')
#plt.fill_between(f,(20*np.log10(np.abs(munc.get_value(dut))-np.abs(munc.get_stdunc(dut)))), (20*np.log10(np.abs(munc.get_value(dut))+np.abs(munc.get_stdunc(dut)))), color='grey', alpha=0.5, label='uncertainty')
plt.fill_between(f,(20*np.log10(np.abs(munc.get_value(dut_ref))-np.abs(munc.get_stdunc(dut_ref)))), (20*np.log10(np.abs(munc.get_value(dut_ref))+np.abs(munc.get_stdunc(dut_ref)))), color='grey', alpha=0.5, label='$\pm\sigma$ Reference')
plt.legend()
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude (dB)')
plt.xticks([8.5e9, 9.5e9, 10.5e9, 11.5e9, 12.5e9], ['8.5', '9.5', '10.5', '11.5', '12.5'])
plt.grid()
plt.tight_layout()
plt.savefig('UNC.svg')
plt.show()
'''
