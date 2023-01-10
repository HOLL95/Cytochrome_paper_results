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
import sys
from harmonics_plotter import harmonics
from multiplotter import multiplot
import os
import math
import copy
import pints
import time
from single_e_class_unified import single_electron
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,zoomed_inset_axes, mark_inset
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import itertools
from PIL import Image
def empty(arg):
    return arg
rowspan=16
col_space=5


#inset_axes = inset_axes(mega_ax,
#                        width="30%", # width = 30% of parent_bbox
#                        height=0.3, # height : 1 inch
#                        loc=1)
example_harms=6
height =0.1
start_harm=1
harm_range=list(range(start_harm, start_harm+example_harms))
num_harms=len(harm_range)
SV_example_ax=[]
ramped_example_ax=[]
figure=multiplot(3,4, **{"harmonic_position":[3], "num_harmonics":num_harms, "orientation":"portrait",  "plot_width":rowspan,"plot_height":2, "row_spacing":2,"col_spacing":col_space, "plot_height":1, "harmonic_spacing":2, "font_size":14})
print(figure.axes_dict)
grid=figure.gridspec
mega_loc=(14,60)
subplotspec=grid.new_subplotspec(mega_loc,(num_harms) , (rowspan))
legend_ax=plt.subplot(subplotspec)
legend_ax.set_axis_off()
#legend_loc=(21,0)
#subplotspec_leg=grid.new_subplotspec(legend_loc,(num_harms) , (rowspan))
#legend_ax=plt.subplot(subplotspec_leg)
#legend_ax.set_axis_off()
mega_ax=figure.axes_dict["col3"][-1]
trumpet_dict=np.load("Experimental_data/DCV/Trumpet_data.npy", allow_pickle=True).item()
colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(1, 3):
    mega_ax.scatter(trumpet_dict["xaxis"], trumpet_dict["yaxis_"+str(i)], color=colors[0], s=5)
    print([round(10**x, 3) for x in trumpet_dict["xaxis"]])
mega_ax.set_xlabel("Log(scan rate (mV))")
mega_ax.set_ylabel("Peak position (V)")
font_size=20


#mega_ax.text( -0.1, 1.1,"G", transform=mega_ax.transAxes,
            #size=font_size, weight='bold')

#plt.show()

letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letter_list=letters.split()
letter_counter=-1
for col in figure.axes_dict.keys():
    letter_step=1
    #letter_counter=-1#int(col[-1])-1-letter_step
    #print(letter_counter)
    col4_flag=False
    if col=="col4":
        col4_flag=True
        step=num_harms
        start=0
    else:
        step=1
        start=0

    for i in range(start, len(figure.axes_dict[col]), step):
        letter_counter+=1
        print(letter_counter)
        if col4_flag==True:
            figure.axes_dict[col][i].text( -0.2, 1.6,letters[letter_counter], transform=figure.axes_dict[col][i].transAxes,
                        size=font_size, weight='bold')
        else:
            figure.axes_dict[col][i].text( -0.2, 1.1,letters[letter_counter], transform=figure.axes_dict[col][i].transAxes,
                        size=font_size, weight='bold')

#plt.show()

fig=plt.gcf()
fig.set_size_inches(14, 9)
#mega_ax.arrow(0,-0.01,0,1.01,head_width=0.05, head_length=0.1, width=0.02, alpha=0.6, color="green")# , transform=mega_ax.transAxes )

print(figure.axes_dict)


data_dict=dict(zip(["DCV_1", "ramped", "SV"], [0,1, 2]))
SV_param_list={
    "E_0":-0.2,
    'E_start':  -0.34024, #(starting dc voltage - V)
    'E_reverse':0.2610531,
    'omega':9.015120071612014, #8.88480830076,  #    (frequency Hz)
    "original_omega":9.015120071612014,
    'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl':0, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 10, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "cap_phase":3*math.pi/2,
    'sampling_freq' : (1.0/400),
    'phase' :3*math.pi/2,
    "time_end": -1,
    'num_peaks': 25,
}
ramped_param_list=copy.deepcopy(SV_param_list)
directory=os.getcwd()
dir_list=directory.split("/")
exp_loc=os.getcwd()+"/Experimental_data/"
del ramped_param_list["original_omega"]
del ramped_param_list["num_peaks"]
dcv_param_list=copy.deepcopy(ramped_param_list)
ramped_dif_params={'E_start':  -0.33898177567074794, 'E_reverse':0.26049326614698887,'omega':8.959294996508683,"v":0.022316752195354346,'d_E': 150*1e-3,"phase":0}
dcv_dif_params={'E_start':  -0.39, 'E_reverse':0.3,'omega':8.94,"v":30*1e-3,'d_E': 0,"phase":0}
dcv_keys=dcv_dif_params.keys()
for key in ramped_dif_params.keys():
    ramped_param_list[key]=ramped_dif_params[key]
    if key in dcv_keys:
        dcv_param_list[key]=dcv_dif_params[key]
SV_vals=[[-0.007311799334716082, 0.06011218461168426, 100.47324948026552, 49.49286763913283, 0.00011151822964084257, 0.00033546831322842086, 0.009999941792246102,0, 2.999999994953957e-11, 9.015005507706533, 6.283185267250843, 1.9462916200793343, 0.4847309255453626],
        [-0.07133935323836932, 0.04433940419884379, 217.58306150192792, 135.0495596023161, 9.62463515705647e-06, 0.01730398477905308, 0.04999871276633058, -0.0007206743270165433, 1.37095896576959e-11, 9.01499164308653, 4.7220768639743085, 4.554136092744141, 0.5999999989106146]
]
experiment_dict={
                "ramped":{"file_loc":"FTACV/2_11_2020", "filename":"FTACV_Cyt_1_cv_","plot_loc":2, "decimation":32, "method":"ramped", "params":ramped_param_list, "transient":1/SV_param_list["omega"], "bounds":20,"harm_range":list(range(3, 8)),
                            "values":[-0.06489530855044306, 0.025901449972125516, 48.1736028007526, 68.30159798782522, 4.714982669828995e-05, 0.04335254198388333*0, -0.004699728058449013*0, 2.1898117688472174e-11, 8.959294458587753,0, 0.5592126258301378],
                            "param_list":["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2","gamma","omega","phase", "alpha"],
                    },
                "SV":{"file_loc":"PSV/2_11_2020", "filename":"PSV_Cyt_1_cv_","plot_loc":4, "decimation":16, "method":"sinusoidal", "params":SV_param_list, "transient":1/ramped_param_list["omega"],"bounds":20000,"harm_range":list(range(4, 9)),
                    "values":SV_vals[1],#[-0.07963988256472493, 0.043023349442016245, 20.550878790990733, 581.5147052074157, 8.806259570340898e-05, -0.045583168350011485, -0.00011159862236990309, 0.00018619134662841048, 2.9947102043021914e-11, 9.014976375142606, 5.699844468024501, 5.18463541959069, 0.5999994350046962],
                    "param_list":["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"],
                    },
                "DCV_1":{"file_loc":"DCV/2_11_2020", "filename":"dcV_Cjx-183D_WT_pH_7_1_3","plot_loc":1, "method":"dcv", "params":dcv_param_list, "transient":False,"bounds":2000,"harm_range":list(range(4, 9)),
                    "values":[-0.051002951188454194, 1.0000168463482572e-05, 0.5871952270775241, 3.4006592554802374e-07, 3.107639506959287e-11, 0.4521221354084581],
                    "param_list":["E0_mean", "E0_std","k_0","Ru","gamma", "alpha"],
                    },
                "DCV_2":{"file_loc":"DCV/2_11_2020", "filename":"dcV_Cjx-183D_WT_pH_7_2_3","plot_loc":3, "method":"dcv", "params":dcv_param_list, "transient":False,"bounds":2000,"harm_range":list(range(4, 9)),
                    "values":[-0.049697063448006555, 0.0037230268328498823, 0.3527926751864493, 4.773196671248675e-10, 1.196896571851427e-11, 0.5513930594926286],
                    "param_list":["E0_mean", "E0_std","k_0","Ru","gamma", "alpha"],
                    },
                "ramped_blank":{"file_loc":"FTACV/2_11_2020", "filename":"FTacV_Blank_1_dec_cv_current", "decimation":None, "params":[],"transient":1/ramped_param_list["omega"],
                    },
                "SV_blank":{"file_loc":"PSV/2_11_2020", "filename":"PSV_Blank_ 1_dec_cv_current", "decimation":None,"params":[],"transient":1/SV_param_list["omega"],
                    },
                "DCV_1_blank":{"file_loc":"DCV/2_11_2020", "filename":"dcV_Blank_1_1", "decimation":None,"params":[],"transient":False,
                    },
                }
unchanged_params=["omega"]
plot_keys=["ramped", "SV","DCV_1", ]
harmonic_files=["ramped", "SV"]
master_simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":[16],
    "GH_quadrature":True,
    "test": False,
    "method": None,
    "phase_only":False,
    "likelihood":"timeseries",
    "numerical_method": "Brent minimisation",
    "label": "MCMC",
    "optim_list":[]
}

master_other_values={
    "filter_val": 0.5,
    "harmonic_range":harm_range,
    "experiment_time": None,
    "experiment_current": None,
    "experiment_voltage":None,
    "bounds_val":20000,
}
master_param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[5,15],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,2e-3], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [1e-12,1e-9],
    'k_0': [0, 2e2], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[-0.08, 0.04],
    "E0_std": [1e-4,  0.1],
    'phase' : [math.pi, 2*math.pi],
}

import matplotlib.ticker as mtick
experiment_counter=0
for experiment_type in plot_keys:
    for blank in ["Modified", "Blank"]:
        data_loc=exp_loc+experiment_dict[experiment_type]["file_loc"]
        file_name=experiment_dict[experiment_type]["filename"]
        params=experiment_dict[experiment_type]["params"]
        if experiment_type in harmonic_files:
            dec_amount=experiment_dict[experiment_type]["decimation"]
            voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
            if blank=="Blank":
                data_loc=exp_loc+experiment_dict[experiment_type+"_blank"]["file_loc"]
                file_name=experiment_dict[experiment_type+"_blank"]["filename"]
                current_data_file=np.loadtxt(data_loc+"/"+file_name)
                current_data=current_data_file[:,1]
                time_data=current_data_file[:,0]
                volt_data=voltage_data_file[0::64, 1]
            else:
                current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
                volt_data=voltage_data_file[0::dec_amount, 1]
                time_data=current_data_file[0::dec_amount, 0]
                current_data=current_data_file[0::dec_amount, 1]

            del current_data_file
            del voltage_data_file
        else:
            if blank=="Blank":
                data_loc=exp_loc+experiment_dict[experiment_type+"_blank"]["file_loc"]
                file_name=experiment_dict[experiment_type+"_blank"]["filename"]
            dcv_file=np.loadtxt(data_loc+"/"+file_name, skiprows=2)
            time_data=dcv_file[:,0]
            volt_data=dcv_file[:,1]
            current_data=dcv_file[:,2]
        experiment_other_vals=copy.deepcopy(master_other_values)
        experiment_other_vals["experiment_current"]=current_data
        experiment_other_vals["experiment_time"]=time_data
        experiment_other_vals["experiment_voltage"]=volt_data
        experiment_simulation_options=copy.deepcopy(master_simulation_options)
        experiment_simulation_options["method"]=experiment_dict[experiment_type]["method"]
        experiment_simulation_options["no_transient"]=experiment_dict[experiment_type]["transient"]
        experiment_class=single_electron(None, params, experiment_simulation_options, experiment_other_vals, master_param_bounds)
        time_results=experiment_class.t_nondim(experiment_class.other_values["experiment_time"])
        current_results=experiment_class.i_nondim(experiment_class.other_values["experiment_current"])*1e3
        voltage_results=experiment_class.e_nondim(experiment_class.other_values["experiment_voltage"])
        if experiment_type=="ramped":
            xaxis=time_results
            plot_func=abs
            hanning=True
            xlabel="Time(s)"
        else:
            xaxis=voltage_results
            plot_func=np.real
            hanning=False
            xlabel="Potential(V)"
        explain_axes=figure.axes_dict["col4"]
        if experiment_type in harmonic_files:
            print(experiment_dict[experiment_type]["values"][experiment_dict[experiment_type]["param_list"].index("omega")])
            h_class=harmonics(harm_range, experiment_dict[experiment_type]["values"][experiment_dict[experiment_type]["param_list"].index("omega")], 0.05)

            if blank=="Blank":
                color="red"
            else:
                color=None
            plot_harmonics=h_class.generate_harmonics(time_results, current_results*1e3, hanning=hanning)
            f_idx=tuple(np.where((h_class.f>0) & (h_class.f<(h_class.input_frequency*(start_harm+example_harms+0.5)))))
            figure.axes_dict["col3"][experiment_counter].semilogy(h_class.f[f_idx], abs(h_class.Y[f_idx]), label=blank, color=color)
            figure.axes_dict["col3"][experiment_counter].set_xlabel("Frequency(Hz)")
            figure.axes_dict["col3"][experiment_counter].set_ylabel("Magnitude")

            for i in range(0, len(plot_harmonics)):
                plot_idx=(experiment_counter*num_harms)+i
                #explain_axes[plot_idx].yaxis.set_major_locator(plt.MaxNLocator(2))
                print(plot_idx, len(explain_axes))
                explain_axes[plot_idx].plot(xaxis, plot_func(plot_harmonics[i,:]), color=color)
                for item in (explain_axes[plot_idx].get_xticklabels() + explain_axes[plot_idx].get_yticklabels()):
                        item.set_fontsize(10)
                twinx=explain_axes[plot_idx].twinx()
                twinx.set_ylabel(h_class.harmonics[i], rotation=0)
                twinx.set_yticks([])
                twinx.tick_params(axis="y",direction="in", pad=-22)

                if blank=="Blank":
                    ticks=explain_axes[plot_idx].get_yticks()
                    explain_axes[plot_idx].set_yticks([ticks[1], ticks[-2]])

                #print(ticks)
                if i==num_harms-1:
                    explain_axes[plot_idx].set_xlabel(xlabel)
                if i==num_harms//2:
                    explain_axes[plot_idx].set_ylabel("Current($\\mu A$)")
            plot_dict={
                        "E":{"xaxis":time_results, "yaxis":voltage_results,"xlabel":"Time(s)", "ylabel":"Potential(V)", 'color':"black", "inset":True},
                            "I":{"xaxis":xaxis, "yaxis":current_results,"xlabel":xlabel, "ylabel":"Total current(mA)","color":None, "inset":False}}
        else:
            #figure.axes_dict["row3"][experiment_counter].set_axis_off()
            for i in range(0, num_harms):
                explain_axes[(experiment_counter*num_harms)+i].set_axis_off()
            plot_dict={
                        "E":{"xaxis":time_results, "yaxis":voltage_results,"xlabel":"Time(s)", "ylabel":"Potential(V)", 'color':"black", "inset":True},
                        "I":{"xaxis":xaxis, "yaxis":current_results*1e3,"xlabel":xlabel, "ylabel":"Total current($\\mu$A)","color":None, "inset":False}}
        plots=["E", "I"]
        experiment=[ "r-FTACV","PSV","DCV", ]
        for j in range(0, 2):
            col="col{0}".format(j+1)
            if j==0 and blank=="Blank":
                continue
            if j==0:
                if len(experiment[experiment_counter])==3:
                #figure.axes_dict[col][experiment_counter].set_title(experiment[experiment_counter])
                    figure.axes_dict[col][experiment_counter].text( -0.5, 0.4,experiment[experiment_counter], transform=figure.axes_dict[col][experiment_counter].transAxes,
                                size=18, weight='bold', rotation="vertical")
                else:
                    figure.axes_dict[col][experiment_counter].text( -0.5, 0.25,experiment[experiment_counter], transform=figure.axes_dict[col][experiment_counter].transAxes,
                                size=18, weight='bold', rotation="vertical")
            if blank=="Blank":
                color="red"
            else:
                color=plot_dict[plots[j]]["color"]

            figure.axes_dict[col][experiment_counter].plot(plot_dict[plots[j]]["xaxis"], plot_dict[plots[j]]["yaxis"], color=color)
            if plot_dict[plots[j]]["inset"]==True and experiment_type=="ramped":
                inset_len=22
                y1=-0.5
                y_height=0.35
                x1=time_results[len(time_results)//2]-(inset_len/2)
                #axins = zoomed_inset_axes(figure.axes_dict[col][experiment_counter], 0.5, loc="center", bbox_to_anchor=(0.5,0.25), bbox_transform=figure.axes_dict[col][experiment_counter].transAxes)
                axins=figure.axes_dict[col][experiment_counter].inset_axes([x1, y1, inset_len, y_height], transform=figure.axes_dict[col][experiment_counter].transData)
                axins.spines['bottom'].set_color('lightslategrey')
                axins.spines['top'].set_color('lightslategrey')
                axins.spines['right'].set_color('lightslategrey')
                axins.spines['left'].set_color('lightslategrey')
                axins.set_xticks([])
                axins.set_yticks([])
                #mark_inset(figure.axes_dict[col][experiment_counter], axins, loc1=1, loc2=2, fc="none", ec="0.5")
                inset_xlim=[2,3.8]
                time_idx=np.where((time_results>inset_xlim[0]) & (time_results<inset_xlim[1]))
                ramped_potential=voltage_results[time_idx]
                inset_ylim=[min(ramped_potential), max(ramped_potential)]
                box_points=[[x1, y1], [x1, y1+y_height]]
                for p in range(0, 2):
                    index_1=[0]*4
                    index_2=[1]*4
                    index_1[-(p+1)]=1
                    index_2[p]=0
                    indices=[index_1, index_2]
                    for z in range(0, 2):
                        figure.axes_dict[col][experiment_counter].plot([inset_xlim[indices[z][0]], inset_xlim[indices[z][2]]], [inset_ylim[indices[z][1]], inset_ylim[indices[z][3]]], color="lightslategrey")
                figure.axes_dict[col][experiment_counter].plot([inset_xlim[1], box_points[0][0]], [inset_ylim[0], box_points[0][1]], color="lightslategrey")
                figure.axes_dict[col][experiment_counter].plot([inset_xlim[1], box_points[1][0]], [inset_ylim[1], box_points[1][1]], color="lightslategrey")
                ramped_time=time_results[time_idx]
                axins.plot(ramped_time, ramped_potential, color=color)



            figure.axes_dict[col][experiment_counter].set_xlabel(plot_dict[plots[j]]["xlabel"])
            figure.axes_dict[col][experiment_counter].set_ylabel(plot_dict[plots[j]]["ylabel"])
    experiment_counter+=1
#figure.axes_dict["row3"][1].legend(loc="upper right")
#data_dict[experiment_type][1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
plt.subplots_adjust(top=0.950,
bottom=0.065,
left=0.09,
right=0.98,
wspace=0.7, hspace=0.3)

#plt.show()
legend_ax.plot(0, 0, label="Potential input", color="black")
legend_ax.plot(0, 0, label="$\it{Cj}$X183")
legend_ax.plot(0,0, label="Blank", color="red")
legend_ax.legend(loc="center", bbox_to_anchor=(0.5, 0.5))

plt.show()
save_path="mega_fig.png"
fig.savefig(save_path, dpi=500)
img = Image.open(save_path)
basewidth = float(img.size[0])//2
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((int(basewidth),hsize), Image.ANTIALIAS)
img.save(save_path, "PNG", quality=95, dpi=(500, 500))
