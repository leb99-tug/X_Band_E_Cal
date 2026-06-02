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

#Initialize waveguide and standards


ref_short = rf.Network('RADAR_CAL/RadarCal/Ver_0.s1p')


ref_short_l8 = rf.Network('RADAR_CAL/RadarCal/Ver_375.s1p')
ref_short_l4 = rf.Network('RADAR_CAL/RadarCal/Ver_750.s1p')
ref_short_3l8 = rf.Network('RADAR_CAL/RadarCal/Ver_1125.s1p')

meas_short = rf.Network('RADAR_CAL/RadarCal/plane_calkit.s1p')

meas_short_1 = rf.Network('RADAR_CAL/CalKit_MeasurementData/messung_pos_3653.s1p')
meas_short_2 = rf.Network('RADAR_CAL/CalKit_MeasurementData/messung_pos_3357.s1p')
meas_short_3 = rf.Network('RADAR_CAL/CalKit_MeasurementData/messung_pos_3046.s1p')

match_std = rf.Network('RADAR_CAL/CalKit_MeasurementData/messung_pos_2058.s1p')

freq = ref_short.frequency
f = freq.f
ideal_short_rho = -1

WR90 = rf.RectangularWaveguide(freq,a=22.86E-3,z0=50 )
WR90_short = WR90.short()
rho1 = WR90.line(3.75, 'mm')**WR90_short
rho2 = WR90.line(7.5, 'mm')**WR90_short
rho3 = WR90.line(11.25, 'mm')**WR90_short



cal = xBand_ECal(standard1=meas_short_1.s11, standard2=meas_short_2.s11, standard3=meas_short_3.s11, rho1=rho1, rho2=rho2, rho3=rho3, 
                sigma_NF=(0.01**2)*np.ones(len(freq)), sigma_NT=(0.01**2)*np.ones(len(freq)), sigma_L=(0.01**2)*np.ones(len(freq)),
                sigma_DD=(0.01**2)*np.ones(len(freq)), sigma_DT=(0.01**2)*np.ones(len(freq)), sigma_DM=(0.01**2)*np.ones(len(freq)),
                sigma_RR=(0.01**2)*np.ones(len(freq)), sigma_RT=(0.01**2)*np.ones(len(freq)), sigma_RM=(0.01**2)*np.ones(len(freq)),
                sigma_SR=(0.01**2)*np.ones(len(freq)), find_lengths=True, find_lengths_options="de+nm",
                enhanced_console_output=True,initial_guess=[ 2.28151833 , 4.65532912 ,10.09979123], ref_standard=meas_short.s11, ref_standard_rho=WR90_short, Waveguide=WR90)

ref_cal = xBand_ECal(standard1=ref_short_l8.s11, standard2=ref_short_l4.s11, standard3=ref_short_3l8.s11, rho1=rho1, rho2=rho2, rho3=rho3, 
                sigma_NF=(0.01**2)*np.ones(len(freq)), sigma_NT=(0.01**2)*np.ones(len(freq)), sigma_L=(0.01**2)*np.ones(len(freq)),
                sigma_DD=(0.01**2)*np.ones(len(freq)), sigma_DT=(0.01**2)*np.ones(len(freq)), sigma_DM=(0.01**2)*np.ones(len(freq)),
                sigma_RR=(0.01**2)*np.ones(len(freq)), sigma_RT=(0.01**2)*np.ones(len(freq)), sigma_RM=(0.01**2)*np.ones(len(freq)),
                sigma_SR=(0.01**2)*np.ones(len(freq)), find_lengths=True, find_lengths_options="de+nm",
                enhanced_console_output=True,initial_guess=[2.65, 7.34, 11.25], ref_standard=ref_short.s11, ref_standard_rho=WR90_short, Waveguide=WR90)


#run E-Calibration
cal.run()
ref_cal.run()


dut_ref = ref_cal.apply_cal(match_std.s11)
dut = cal.apply_cal(match_std.s11)

plt.plot(f,(20*np.log10(np.abs(munc.get_value(dut)))), label='de-embedded')
plt.plot(f,(20*np.log10(np.abs(munc.get_value(dut_ref)))), label='reference')
#plt.fill_between(f,(20*np.log10(np.abs(munc.get_value(dut))-np.abs(munc.get_stdunc(dut)))), (20*np.log10(np.abs(munc.get_value(dut))+np.abs(munc.get_stdunc(dut)))), color='grey', alpha=0.5, label='uncertainty')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude of Reflection Coefficient (dB)')
plt.grid()
plt.show()