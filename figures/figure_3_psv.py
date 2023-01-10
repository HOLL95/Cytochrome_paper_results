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
import copy
import pints
import time
from multiplotter import multiplot
from single_e_class_unified import single_electron
from scipy.integrate import odeint
import matplotlib.ticker as mtick
data_loc=os.getcwd()
locations=["Experimental_data/PSV/2_11_2020","Experimental_data/PSV/1_2_2021","Experimental_data/PSV/15_2_2021"]
names=["PSV_Cyt_1_", "Cyt_9_hz_1_", "9_Hz_1_"]
files=os.listdir(data_loc)
experimental_dict={}
harm_list=list(range(4, 11))

figure=multiplot(2,3, **{"harmonic_position":[1], "num_harmonics":len(harm_list), "orientation":"landscape",  "plot_width":5,"plot_height":3, "row_spacing":1,"col_spacing":1, "plot_height":1, "harmonic_spacing":1, "font_size":9})
print(figure.axes_dict)
params_25=[
[-0.07132938220428349, 0.044300587374493786, 216.10078804184, 136.2162023875664, 9.618031445251783e-06, 0.01714663192094401, 0.049780519830921854, -0.0007213188392024977, 1.3714882304886588e-11, 9.014991776457336, 4.722403304312193, 4.554851186125063, 0.5999999899544305],
[-0.06813418210437491, 0.05620641891576498, 377.19958608605845, 83.16409435426613, 9.999899522062466e-06, 0.15887567880177011, 0.05268229708793217, -0.0005329278136488938, 2.059535183842538e-11, 9.014974754447994, 4.642166107758895, 4.520368351387462, 0.5999999851122721],
[-0.06498696550965283, 0.04920932475992681, 185.0913243030523, 101.70578388805512, 9.998005534750987e-06, 0.09791298173671645, 0.04578563043034162, -0.000471747249437271, 1.8430279223839692e-11, 9.015421085908832, 4.709656633696859, 4.628222719686389, 0.5999999987849658],
]
params_5=[
    [-0.07166835210313773, 0.04508238949075683, 173.8086034279432, 148.68802663242204, 9.79381671170001e-06, 0.013604126711432143, 0.039814752589476435, -0.0005607024918967279, 1.3486810794934256e-11, 9.014989313901726, 4.729029409373901, 4.572432373365053, 0.5999999978437339],
    [-0.06784206521363831, 0.05288416568757893, 176.4806332814906, 316.8115851855736, 9.999974524355597e-06, 0.07945740018896924, 0.02144764262225811, -0.00043813505754695253, 2.0468416218511848e-11, 9.014997636697508, 4.695646546457142, 4.599353368087924, 0.5999999969688337],
    [-0.0653848527549318, 0.05067559053083178, 172.89404940953966, 81.52528235471596, 9.999113934162367e-06, 0.09544156388150249, 0.0453910470182763, -0.00037721164932495027, 1.7920567448283947e-11, 9.01542111961031, 4.711676000062636, 4.628290457595525, 0.5999999952723338]
    ]
params=params_5
#inferred_params=
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
omega_pos=9
dec_amount=32
def one_tail(series):
    if len(series)%2==0:
        return series[:len(series)//2]
    else:
        return series[:len(series)//2+1]
letters=["A", "B", "C", ]
for i in range(0, len(locations)):
    file_name=("/").join([data_loc,locations[i],names[i]])
    current_data_file=np.loadtxt(file_name+"cv_current")
    voltage_data_file=np.loadtxt(file_name+"cv_voltage")
    volt_data=voltage_data_file[0::dec_amount, 1]
    param_list={
        "E_0":-0.2,
        'E_start':  min(volt_data[len(volt_data)//4:3*len(volt_data)//4]), #(starting dc voltage - V)
        'E_reverse':max(volt_data[len(volt_data)//4:3*len(volt_data)//4]),
        'omega':9.015120071612014, #8.88480830076,  #    (frequency Hz)
        "original_omega":params[i][omega_pos],
        'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
        "cap_phase":3*math.pi/2,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :3*math.pi/2,
        "time_end": -1,
        'num_peaks': 30,
    }
    print(param_list)
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=2/(param_list["original_omega"])
    simulation_options={
        "no_transient":time_start,
        "numerical_debugging": False,
        "experimental_fitting":True,
        "dispersion":False,
        "dispersion_bins":[30],
        "GH_quadrature":True,
        "hanning":False,
        "test": False,
        "method": "sinusoidal",
        "phase_only":False,
        "likelihood":likelihood_options[1],
        "numerical_method": solver_list[1],
        "label": "MCMC",
        "optim_list":[]
    }

    other_values={
        "filter_val": 0.5,
        "harmonic_range":list(range(4,100,1)),
        "experiment_time": current_data_file[0::dec_amount, 0],
        "experiment_current": current_data_file[0::dec_amount, 1],
        "experiment_voltage":volt_data,
        "bounds_val":200000,
    }
    param_bounds={
        'E_0':[-0.1, 0.1],
        'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
        'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
        'Cdl': [0,1e-4], #(capacitance parameters)
        'CdlE1': [-0.1,0.1],#0.000653657774506,
        'CdlE2': [-0.05,0.05],#0.000245772700637,
        'CdlE3': [-0.05,0.05],#1.10053945995e-06,
        'gamma': [0.1*param_list["original_gamma"],8*param_list["original_gamma"]],
        'k_0': [50, 1e3], #(reaction rate s-1)
        'alpha': [0.4, 0.6],
        "cap_phase":[math.pi/2, 2*math.pi],
        "E0_mean":[-0.1, -0.04],
        "E0_std": [1e-4,  0.1],
        "E0_skew": [-10, 10],
        "alpha_mean":[0.4, 0.65],
        "alpha_std":[1e-3, 0.3],
        "k0_shape":[0,1],
        "k0_scale":[0,1e4],
        'phase' : [math.pi, 2*math.pi],
    }
    print(param_list["E_start"], param_list["E_reverse"])
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    del current_data_file
    del voltage_data_file
    cyt.define_boundaries(param_bounds)
    time_results=cyt.other_values["experiment_time"]
    current_results=cyt.i_nondim(cyt.other_values["experiment_current"])
    print(current_results[0], current_results[-1])
    voltage_results=cyt.e_nondim(cyt.other_values["experiment_voltage"])

    h_class=harmonics(harm_list, 1, 0.05)

    cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])
    exp_harms=h_class.generate_harmonics(time_results, current_results)
    cmaes_test=cyt.i_nondim(cyt.test_vals(params[i], "timeseries"))
    cyt.simulation_options["top_hat_return"]="inverse"
    filtered_time=np.real(cyt.top_hat_filter(cmaes_test))
    filtered_exp=np.real(cyt.top_hat_filter(current_results))
    #fig, ax=plt.subplots(1,1)

    row="row1"
    ax=figure.axes_dict[row][i]
    ax2=ax.twinx()
    tot_line=h_class.single_oscillation_plot(time_results, current_results*1e3, ax=ax2, colour=colours[3], alpha=0.4, label="Total Exp")
    if i==2:
        ax2.set_ylabel("Total current(mA)")
    sim_line=h_class.single_oscillation_plot(time_results, filtered_exp*1e6, ax=ax, colour=colours[0], label="Exp")
    exp_line=h_class.single_oscillation_plot(time_results, filtered_time*1e6, ax=ax, colour=colours[1], label="Sim")

    if i==0:
        ax.set_ylabel("Filtered current ($\\mu A$)")
    if i==1:
        ax.plot(0, 0, color=colours[3], label="Total")
        #ax.legend(loc="center", bbox_to_anchor=[0.5, -0.25], facecolor=None, frameon=False, ncol=3)#(tot_line, sim_line, exp_line), ("Total Exp", "Sim", "Exp"), loc="center", bbox_to_anchor=[0.5, 0.25])
    #ax.set_xlabel("Oscillation period")
    exp_harms=h_class.generate_harmonics(time_results, current_results*1e6)
    sim_harms=h_class.generate_harmonics(time_results, cmaes_test*1e6)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.tick_params(axis="both",direction="in")
    if i==2:
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    else:
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    for j in range(0, h_class.num_harmonics):

        ax=figure.axes_dict["row2"][j+(i*h_class.num_harmonics)]
        if j==0:
            ax.text(-0.2, 1.25,letters[i], transform=ax.transAxes,
                                size=14, weight='bold')
        if i==0:
            if j==h_class.num_harmonics//2:
                ax.set_ylabel("Current($\\mu A$)")
        if j==h_class.num_harmonics-1:
            ax.set_xlabel("Potential(V)")

        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        ax2=ax.twinx()
        ax2.set_ylabel(h_class.harmonics[j], rotation=0)
        ax2.set_yticks([])
        ax.plot(voltage_results, np.real(exp_harms[j,:]))
        ax.plot(voltage_results, np.real(sim_harms[j,:]))
        #ax.tick_params(direction="in")
        ticks=ax.get_yticks()
        print(ticks)
        ax.set_yticks([ticks[1], ticks[-2]])

fig=plt.gcf()
fig.set_size_inches(7, 4.5)
plt.subplots_adjust(top=0.985,
bottom=0.09,
left=0.09,
right=0.98,
hspace=0.25,
wspace=0.8)
plt.show()
save_path="PSV_experiments.png"
fig.savefig(save_path, dpi=500)