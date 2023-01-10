import sys
import os
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="Cytochrome_paper_results"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import numpy as np
import matplotlib.pyplot as plt
from harmonics_plotter import harmonics
import math
from multiplotter import multiplot
import copy
import pints
import time
import matplotlib.ticker as mtick
from single_e_class_unified import single_electron
from scipy.integrate import odeint
data_loc=os.getcwd()
files=os.listdir(data_loc)
experimental_dict={}
harm_list=list(range(4,8,1))
figure=multiplot(2,3, **{"harmonic_position":[0,1, 2], "num_harmonics":len(harm_list), "orientation":"portrait", "plot_width":5,"plot_height":3, "row_spacing":1,"col_spacing":1, "plot_height":1, "harmonic_spacing":1, "font_size":9})
print(figure.axes_dict)
locations=["Experimental_data/FTACV/2_11_2020","Experimental_data/FTACV/1_2_2021","Experimental_data/FTACV/15_2_2021"]
names=["FTACV_Cyt_1_cv_", "cyt_FTACV_1_cv_","FTacV_after_PSV_cv_" ]
dec_amount=64

"""
25 deg version
SV_params=[
[-0.07133935323836932, 0.04433940419884379, 217.58306150192792, 135.0495596023161, 9.62463515705647e-06, 0.01730398477905308, 0.04999871276633058, -0.0007206743270165433, 1.37095896576959e-11, 8.959294458587753, 0, 0.5999999989106146],
[-0.0679798604711702, 0.05773715221549862, 441.4905003376297, 79.60373721578735, 9.999999802298137e-06, 0.09999996707059008, 0.04999999378833646, -0.0005225327690214776, 2.1389331353555487e-11,8.754536606208616, 0, 0.5999999424318273],
[-0.06498696550965283, 0.04920932475992681, 185.0913243030523, 101.70578388805512, 9.998005534750987e-06, 0.09791298173671645, 0.04578563043034162, -0.000471747249437271, 1.8430279223839692e-11, 8.828916016900438, 0,  0.5999999987849658]
]

ramped_params=[
[-0.06141311576695449, 0.030709900003729994,217.58306150192792, 135.0495596023161, 9.62463515705647e-06, 0.01730398477905308, 0.04999871276633058, -0.0007206743270165433, 1.5331633705397376e-11, 8.959294458587753, 0, 0.5999999989106146],
[-0.06303886080464444, 0.0364289044665454, 441.4905003376297, 79.60373721578735, 9.999999802298137e-06, 0.09999996707059008, 0.04999999378833646, -0.0005225327690214776, 1.767417945896251e-11,  8.754536606208616, 0, 0.5999999424318273],
[-0.06292117857959934, 0.03545760947247232, 185.0913243030523, 101.70578388805512, 9.998005534750987e-06, 0.09791298173671645, 0.04578563043034162, -0.000471747249437271, 1.74e-11, 8.828916016900438, 0,  0.5999999987849658]
]"""

SV_params=[
    [-0.07166835210313773, 0.04508238949075683, 173.8086034279432, 148.68802663242204, 9.79381671170001e-06, 0.013604126711432143, 0.039814752589476435, -0.0005607024918967279, 1.3486810794934256e-11,8.959294458587753, 0, 0.5999999989106146],
    [-0.06784206521363831, 0.05288416568757893, 176.4806332814906, 316.8115851855736, 9.999974524355597e-06, 0.07945740018896924, 0.02144764262225811, -0.00043813505754695253, 2.0468416218511848e-11, 8.754536606208616, 0, 0.5999999424318273],
    [-0.0653848527549318, 0.05067559053083178, 172.89404940953966, 81.52528235471596, 9.999113934162367e-06, 0.09544156388150249, 0.0453910470182763, -0.00037721164932495027, 1.7920567448283947e-11, 8.828916016900438, 0,  0.5999999987849658],
    ]
ramped_params=[
    [-0.06186319709065167, 0.03255856563514606,  173.8086034279432, 148.68802663242204, 9.999137732744101e-06, 0.013604126711432143, 0.039814752589476435, -0.0005607024918967279,  1.6836574061827644e-11, 8.959294458587753, 0, 0.5999999989106146],
    [-0.06322562634058088, 0.03586525312400956, 176.4806332814906, 316.8115851855736, 9.999974524355597e-06, 0.07945740018896924, 0.02144764262225811, -0.00043813505754695253,  1.8347679231881322e-11, 8.754403239342299, 0, 0.5999999424318273],
    [-0.06118805085850972, 0.03531752851071228, 172.89404940953966, 81.52528235471596, 9.999113934162367e-06, 0.09544156388150249, 0.0453910470182763, -0.00037721164932495027, 1.4491393361621824e-11, 8.828916016900438, 0,  0.5999999987849658],
    ]
def one_tail(series):
    if len(series)%2==0:
        return series[:len(series)//2]
    else:
        return series[:len(series)//2+1]
letters=["D","E","F","G","H","I"]
for i in range(0, 3):
    loc=data_loc+"/"+locations[i]
    file_name=names[i]
    current_data_file=np.loadtxt(loc+"/"+file_name+"current")
    voltage_data_file=np.loadtxt(loc+"/"+file_name+"voltage")
    volt_data=voltage_data_file[0::dec_amount, 1]
    param_list={
        "E_0":-0.2,
        'E_start':  -0.33898177567074794, #(starting dc voltage - V)
        'E_reverse':0.26049326614698887,
        'omega':SV_params[i][9], #8.88480830076,  #    (frequency Hz)
        "v":0.022316752195354346,
        'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 1.0,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,#0.000653657774506,
        'CdlE2': 0,#0.000245772700637,
        "CdlE3":0,
        'gamma': 1e-11,
        "original_gamma":1e-11,        # (surface coverage per unit area)
        'k_0': 10, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "E0_skew":0.2,
        "cap_phase":0,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :6.283185307179562,
        "time_end": -1,
    }
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=2/(param_list["omega"])
    simulation_options={
        "no_transient":False,#time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":[16],
        "GH_quadrature":True,
        "test": False,
        "method": "ramped",
        "phase_only":False,
        "likelihood":likelihood_options[1],
        "numerical_method": solver_list[1],
        "label": "MCMC",
        "optim_list":[]
    }

    other_values={
        "filter_val": 0.5,
        "harmonic_range":harm_list,
        "experiment_time": current_data_file[0::dec_amount, 0],
        "experiment_current": current_data_file[0::dec_amount, 1],
        "experiment_voltage":volt_data,
        "bounds_val":2000,
    }
    param_bounds={
        'E_0':[-0.1, 0.0],
        "E_start":[0.9*param_list["E_start"], 1.1*param_list["E_start"]],
        "E_reverse":[0.9*param_list["E_reverse"], 1.1*param_list["E_reverse"]],
        "v":[0.9*param_list["v"], 1.1*param_list["v"]],
        'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
        'Cdl': [0,2e-3], #(capacitance parameters)
        'CdlE1': [-0.05,0.05],#0.000653657774506,
        'CdlE2': [-0.01,0.01],#0.000245772700637,
        'CdlE3': [-0.01,0.01],#1.10053945995e-06,
        'gamma': [1.3*param_list["original_gamma"],10*param_list["original_gamma"]],
        'k_0': [0.1, 1e3], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[-0.1, 0.0],
        "E0_std": [1e-4,  0.1],
        "E0_skew": [-10, 10],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        'phase' : [0, 2*math.pi],
    }
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    del current_data_file
    del voltage_data_file
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.i_nondim(cyt.other_values["experiment_current"])
    #print(current_results[0], current_results[-1])
    voltage_results=cyt.other_values["experiment_voltage"]
    h_class=harmonics(other_values["harmonic_range"], param_list["omega"]*cyt.nd_param.c_T0, 0.05)
    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","phase", "alpha"])
    PSV_pred_1=cyt.i_nondim(cyt.test_vals(SV_params[i], "timeseries"))
    PSV_pred_2=cyt.i_nondim(cyt.test_vals(ramped_params[i], "timeseries"))
    pred_1_harms=h_class.generate_harmonics(time_results, PSV_pred_1*1e6, hanning=True)
    pred_2_harms=h_class.generate_harmonics(time_results, PSV_pred_2*1e6, hanning=True)
    exp_harms=h_class.generate_harmonics(time_results, current_results*1e6, hanning=True)
    plot_times=cyt.t_nondim(time_results)
    cyt.simulation_options["method"]="dcv"
    dcv_volts=cyt.e_nondim(cyt.define_voltages())
    cyt.simulation_options["method"]="ramped"
    col="col"+str(i+1)
    for j in range(0, h_class.num_harmonics):
        ax1=figure.axes_dict[col][j]
        ax2=figure.axes_dict[col][j+h_class.num_harmonics]
        axes=[ax1, ax2]
        sim_plot=[abs(pred_1_harms[j,:]), abs(pred_2_harms[j,:])]

        for q in range(0, 2):
            if j==0:
                letter_idx=i+(q*len(ramped_params))
                axes[q].text(-0.2, 1.13,letters[letter_idx], transform=axes[q].transAxes,
                                    size=14, weight='bold')

            axes[q].plot(time_results, abs(exp_harms[j,:]))
            axes[q].plot(time_results, sim_plot[q])
            ax2=axes[q].twinx()
            ax2.set_ylabel(h_class.harmonics[j], rotation=0)
            ax2.set_yticks([])
            if j==h_class.num_harmonics-1:
                axes[q].set_xlabel("Time(s)")
            if i==0:
                if j==h_class.num_harmonics//2:
                    axes[q].set_ylabel("Current($\\mu A$)")
            axes[q].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
fig=plt.gcf()
fig.set_size_inches(7, 4.5)
plt.subplots_adjust(top=0.985,
bottom=0.09,
left=0.09,
right=0.98,
hspace=0.25,
wspace=0.8)
plt.show()
fig.savefig("Ramped_comparison.png", dpi=500)
