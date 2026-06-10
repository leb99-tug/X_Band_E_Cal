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

ref_short = rf.Network('090626/Ver_0.s1p')["8.4-12.5GHz"]#["8.4-12GHz"]


ref_short_l8 = rf.Network('090626/Ver_375.s1p')["8.4-12.5GHz"]
ref_short_l4 = rf.Network('090626/Ver_750.s1p')["8.4-12.5GHz"]
ref_short_3l8 = rf.Network('090626/Ver_1125.s1p')["8.4-12.5GHz"]

meas_short = rf.Network('090626/calkit_l0.s1p')["8.4-12.5GHz"]
meas_short_1 = rf.Network('090626/calkit_l8.s1p')["8.4-12.5GHz"]
meas_short_2 = rf.Network('090626/calkit_l4.s1p')["8.4-12.5GHz"]
meas_short_3 = rf.Network('090626/calkit_3l8.s1p')["8.4-12.5GHz"]


target = rf.Network('090626/target175.s1p')["8.4-12.5GHz"]


freq = ref_short.frequency
f = freq.f

WR90 = rf.RectangularWaveguide(freq,a=22.86E-3,z0=50)
WR90_short = rf.Network(s=(-1-0.000000000001j)*np.ones(len(freq)), frequency=freq, z0=50)


rho1_cal = WR90.line(3.79302482, 'mm')**WR90_short
rho2_cal = WR90.line(7.4010261, 'mm')**WR90_short
rho3_cal = WR90.line(11.31874477, 'mm')**WR90_short




cal = xBand_ECal(standard1=meas_short_1.s11, standard2=meas_short_2.s11, standard3=meas_short_3.s11, rho1=rho1_cal, rho2=rho2_cal, rho3=rho3_cal, 
                sigma_NF=(0.000001**2)*np.ones(len(freq)), sigma_NT=(0.000025**2)*np.ones(len(freq)), sigma_L=(0.0012**2)*np.ones(len(freq)),
                sigma_DD=(0.002**2)*np.ones(len(freq)), sigma_DT=(0.0125**2)*np.ones(len(freq)), sigma_DM=(0.025**2)*np.ones(len(freq)),
                sigma_RR=(0.004**2)*np.ones(len(freq)), sigma_RT=(0.0000001**2)*np.ones(len(freq)), sigma_RM=(0.004**2)*np.ones(len(freq)),
                sigma_SR=(0.05**2)*np.ones(len(freq)), find_lengths=False, find_lengths_options="de",
                enhanced_console_output=True,initial_guess=[3.73036, 7.3369 , 11.3219821], ref_standard=meas_short.s11, ref_standard_rho=WR90_short, Waveguide=WR90)

cal.run()




dut = cal.apply_cal(target.s11)


dut = rf.Network(s=munc.get_value(dut), frequency=freq)
target.plot_s_time(label="Uncalibrated")
dut.plot_s_time(label="Calibrated")
xlim = (0, 40)
plt.xlim(xlim)
plt.show()