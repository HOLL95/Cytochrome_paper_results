import sys
import os
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="Cytochrome_paper_results"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import pints
import matplotlib.pyplot as plt
from pints import plot
import numpy as np
import matplotlib.ticker as mtick
from plotting_funcs import plot_funcs
from PIL import Image
file_loc="MCMC/"
file_names=["alice_cyt_2_MCMC_PSV_exp_1"]
image_names=["PSV_{0}_2d_hist.png".format(x) for x in range(1, 2)]
chain_order=["E0_mean", "E0_std", "k_0", "gamma", "cap_phase","Ru", "phase", "CdlE1", "CdlE2", "CdlE3"]
values=[1e3,1e3,1,1e12, 1, 1, 1, 1, 10, 1e3]
multiply_dict=dict(zip(chain_order, values))
desired_params=["E0_mean", "E0_std", "k_0", "Ru","gamma", "phase", "cap_phase"]

burn=7000
trumpet_DCV=np.load("MCMC/trumpet_MCMC_2")
DCV_burn=6000
chains_dict={}
for file in file_names:
    chains=np.load(file_loc+file, "r")
    chains_dict[file]=chains
trumpet_list=["E0_mean", "k_0"]
trumpet_counter=0
PSV_counter=1
unit_dict={
    "E_0": "V",
    'E_start': "V", #(starting dc voltage - V)
    'E_reverse': "V",
    'omega':"Hz",#8.88480830076,  #    (frequency Hz)
    'd_E': "V",   #(ac voltage amplitude - V) freq_range[j],#
    'v': '$V s^{-1}$',   #       (scan rate s^-1)
    'area': '$cm^{2}$', #(electrode surface area cm^2)
    'Ru': "$\\Omega$",  #     (uncompensated resistance ohms)
    'Cdl': "F", #(capacitance parameters)
    'CdlE1': "",#0.000653657774506,
    'CdlE2': "x10",#0.000245772700637,
    'CdlE3': "x10$^3$",#1.10053945995e-06,
    'gamma': 'pmol cm$^{-2}$',
    'k_0': '$s^{-1}$', #(reaction rate s-1)
    'alpha': "",
    'E0_skew':"",
    "E0_mean":"mV",
    "E0_std": "mV",
    "k0_shape":"",
    "sampling_freq":"$s^{-1}$",
    "k0_loc":"",
    "k0_scale":"",
    "cap_phase":"",
    'phase' : "",
    "alpha_mean": "",
    "alpha_std": "",
    "":"",
    "noise":"",
    "error":"$\\mu A$",
}
fancy_names={
    "E_0": '$E^0$',
    'E_start': '$E_{start}$', #(starting dc voltage - V)
    'E_reverse': '$E_{reverse}$',
    'omega':'$\\omega$',#8.88480830076,  #    (frequency Hz)
    'd_E': "$\\Delta E$",   #(ac voltage amplitude - V) freq_range[j],#
    'v': "v",   #       (scan rate s^-1)
    'area': "Area", #(electrode surface area cm^2)
    'Ru': "Ru",  #     (uncompensated resistance ohms)
    'Cdl': "$C_{dl}$", #(capacitance parameters)
    'CdlE1': "$C_{dlE1}$",#0.000653657774506,
    'CdlE2': "$C_{dlE2}$",#0.000245772700637,
    'CdlE3': "$C_{dlE3}$",#1.10053945995e-06,
    'gamma': '$\\Gamma$',
    'E0_skew':"$E^0$ skew",
    'k_0': '$k_0$', #(reaction rate s-1)
    'alpha': "$\\alpha$",
    "E0_mean":"$E^0 \\mu$",
    "E0_std": "$E^0 \\sigma$",
    "cap_phase":"C$_{dl}$ $\\eta$",
    "k0_shape":"$\\log(k^0) \\sigma$",
    "k0_scale":"$\\log(k^0) \\mu$",
    "alpha_mean": "$\\alpha\\mu$",
    "alpha_std": "$\\alpha\\sigma$",
    'phase' : "$\\eta$",
    "sampling_freq":"Sampling rate",
    "":"Experiment",
    "noise":"$\sigma$",
    "error":"RMSE",
}
nbins=15
plot_count=0
def plot_kde_1d(x, ax, num=None, colour=None):
    xmin = np.min(x)
    xmax = np.max(x)
    x1 = np.linspace(xmin, xmax, 100)
    x2 = np.linspace(xmin, xmax, 10)
    ax.hist(x, bins=x2, label=num, color=colour)
def plot_kde_2d(x, y, ax, colour=None):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    ax.scatter(x, y, s=0.5, alpha=0.5, color=colour)
n_param=len(chain_order)

burn =6000
import matplotlib as mpl
mpl.rcParams["font.size"]=14
titles=[fancy_names[x]+"("+unit_dict[x]+")" if unit_dict[x]!="" else fancy_names[x] for x in chain_order]
resize=True
image_counter=0
import copy
empty_array=[0,0,0]
labels=["median", "lower_median", "upper_median", "lower_bound", "upper_bound"]
functions=[np.quantile, np.quantile, np.quantile, np.min, np.max]
args=[[0.5],[0.25], [0.75], [], []]
func_dict={labels[i]:{"function":functions[i], "args":args[i]} for i in range(0, len(labels))}
box_params={param:{x:copy.deepcopy(empty_array) for x in labels} for param in chain_order}
exp_counter=0
for file in file_names:
        fig, ax=plt.subplots(n_param, n_param)
        chain_results=chains_dict[file]
        for i in range(0,n_param):
            pooled_chain_i=[chain_results[x, burn:, i] for x in range(0, 3)]
            chain_i=np.concatenate(pooled_chain_i)
            for m in range(0, len(labels)):
                box_params[chain_order[i]][labels[m]][exp_counter]=func_dict[labels[m]]["function"](chain_i, *func_dict[labels[m]]["args"])
            chain_i=np.multiply(chain_i, values[i])
            for j in range(0, n_param):
                if i==j:
                    axes=ax[i,j]
                    ax1=axes.twinx()
                    plot_kde_1d(chain_i, ax=axes, colour="darkslategrey")
                    ticks=axes.get_yticks()
                    axes.set_yticks([])
                    ax1.set_yticks(ticks)
                elif i<j:
                    ax[i,j].axis('off')
                else:
                    axes=ax[i,j]
                    pooled_chain_j=[chain_results[x, burn:, j] for x in range(0, 3)]
                    chain_j=np.multiply(np.concatenate(pooled_chain_j), values[j])
                    plot_kde_2d(chain_j,chain_i, ax=axes, colour="lightslategrey")
                if i!=0:
                    if chain_order[i]=="CdlE3":
                        ax[i, 0].set_ylabel(titles[i], labelpad=20)
                    elif chain_order[i]=="gamma":
                        ax[i, 0].set_ylabel(titles[i], labelpad=30)
                    else:
                        ax[i, 0].set_ylabel(titles[i])
                if i<n_param-1:
                    ax[i,j].set_xticklabels([])#
                if j>0 and i!=j:
                    ax[i,j].set_yticklabels([])
                if j!=n_param:
                    ax[-1, i].set_xlabel(titles[i])
                    plt.setp( ax[-1, i].xaxis.get_majorticklabels(), rotation=30 )

        plt.subplots_adjust(top=0.98,
                            bottom=0.12,
                            left=0.1,
                            right=0.955,
                            hspace=0.18,
                            wspace=0.155)

        fig = plt.gcf()
        fig.set_size_inches((14,9))
        save_path=image_names[image_counter]
        fig.savefig(save_path, dpi=500)
        if resize==True:
            img = Image.open(save_path)
            basewidth = float(img.size[0])//2
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((int(basewidth),hsize), Image.ANTIALIAS)
            img.save(save_path, "PNG", quality=95, dpi=(500, 500))
        image_counter+=1
        exp_counter+=1
plt.show()

