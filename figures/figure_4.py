import pints
import matplotlib.pyplot as plt
from pints import plot
import numpy as np
import matplotlib.ticker as mtick
file_loc="MCMC/"
file_names=["alice_cyt_2_MCMC_PSV_exp_1", "nick_cyt_0_MCMC_PSV_exp_2", "nick_cyt_0_MCMC_2_PSV_exp_3"]
chain_order=["E0_mean", "E0_std", "k_0", "gamma", "cap_phase","Ru", "phase", "CdlE1", "CdlE2", "CdlE3"]
desired_params=["E0_mean", "E0_std", "k_0", "Ru","gamma", "phase", "cap_phase"]
fig, ax=plt.subplots(2, 4)
burn=6000
trumpet_DCV=np.load(file_loc+"trumpet_MCMC_2")

DCV_burn=6000
def plot_kde_1d(x, ax, num=None):
    xmin = np.min(x)
    xmax = np.max(x)
    x1 = np.linspace(xmin, xmax, 100)
    x2 = np.linspace(xmin, xmax, 10)
    ax.hist(x, bins=x2, label=num)

def plot_kde_2d(x, y, ax):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    ax.scatter(x, y, s=0.5, alpha=0.5)
cheap_dict={}
params=["E_0", "k_0", "dcv_sep", "Ru"]


chains_dict={}

for file in file_names:
    chains=np.load(file_loc+file, "r")
    chains_dict[file]=chains
trumpet_list=["E0_mean", "k_0", "Ru"]
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
    'CdlE2': "",#0.000245772700637,
    'CdlE3': "",#1.10053945995e-06,
    'gamma': 'mol cm^{-2}$',
    'k_0': 's^{-1}$', #(reaction rate s-1)
    'alpha': "",
    'E0_skew':"",
    "E0_mean":"V",
    "E0_std": "V",
    "k0_shape":"",
    "sampling_freq":"$s^{-1}$",
    "k0_loc":"",
    "k0_scale":"",
    "cap_phase":"rads",
    'phase' : "rads",
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
    'gamma': '$\\Gamma',
    'E0_skew':"$E^0$ skew",
    'k_0': '$k_0', #(reaction rate s-1)
    'alpha': "$\\alpha$",
    "E0_mean":"$E^0 \\mu$",
    "E0_std": "$E^0 \\sigma$",
    "cap_phase":"C$_{dl}$ phase",
    "k0_shape":"$\\log(k^0) \\sigma$",
    "k0_scale":"$\\log(k^0) \\mu$",
    "alpha_mean": "$\\alpha\\mu$",
    "alpha_std": "$\\alpha\\sigma$",
    'phase' : "Phase",
    "sampling_freq":"Sampling rate",
    "":"Experiment",
    "noise":"$\sigma$",
    "error":"RMSE",
}
nbins=15
e0_lists=np.zeros((2, 3))
mean_counter=0
std_counter=0
for i in range(0, len(desired_params)):
        plot_idx=i
        curr_ax=ax[plot_idx//4, plot_idx%4]
        for file in file_names:
            chains=chains_dict[file]
            chain_idx=chain_order.index(desired_params[i])
            pooled_chain=[chains[x, burn:, chain_idx] for x in range(0, 3)]
            hist_chain=np.concatenate(pooled_chain)
            if plot_idx==2:
                curr_ax.hist(hist_chain, label="PSV "+str(PSV_counter), weights=np.ones(len(hist_chain))*1/1000, bins=nbins)
                PSV_counter+=1
            else:
                curr_ax.hist(hist_chain,weights=np.ones(len(hist_chain))*1/1000, bins=nbins)
                if "_mean" in desired_params[i]:
                    e0_lists[0, mean_counter]=np.mean(hist_chain)
                    mean_counter+=1
                if "_std" in desired_params[i]:
                    e0_lists[1, std_counter]=np.mean(hist_chain)
                    std_counter+=1

        if desired_params[i] in trumpet_list:
            if desired_params[i]=="Ru":
                trumpet_counter=3
            trumpet_chain=[trumpet_DCV[x, DCV_burn:, trumpet_counter] for x in range(0, 3)]
            trumpet_counter+=1
            trumpet_hist=np.concatenate(trumpet_chain)
            curr_ax.hist(trumpet_hist, label="DCV", weights=np.ones(len(trumpet_hist))*1/1000, alpha=0.5, bins=nbins)
        if plot_idx==2:
            curr_ax.legend(loc="center", bbox_to_anchor=[1.55, -1])
        if plot_idx%4==0:
            curr_ax.set_ylabel("Frequency $10^3$")
        if desired_params[i]=="gamma":
            curr_ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        if unit_dict[desired_params[i]]=="":
            xlabel_unit=""
        else:
            xlabel_unit="({0})".format(unit_dict[desired_params[i]])
        curr_ax.set_xlabel(fancy_names[desired_params[i]]+xlabel_unit)
ax[1, 3].axis("off")
from scipy.stats import norm
fig.set_size_inches(7, 4.5)
plt.subplots_adjust(top=0.955,
bottom=0.11,
left=0.085,
right=0.995,
hspace=0.285,
wspace=0.25)
plt.show()
fig.savefig("PSV_hists.png", dpi=500)
fig, ax=plt.subplots()
print(e0_lists)
for i in range(0, len(e0_lists[0])):
    mean=e0_lists[0, i]
    std=e0_lists[1, i]
    x=np.linspace(norm.ppf(1e-4, loc=mean, scale=std), norm.ppf(1-1e-4, loc=mean, scale=std),1000 )
    y=np.zeros(len(x))
    y[0]=0#1e-4#norm.cdf(x[0], loc=mean, scale=std)-norm.cdf(x[0]*2, loc=mean, scale=std)
    for j in range(1, len(y)):
        y[j]=norm.cdf(x[j], loc=mean, scale=std)-norm.cdf(x[j-1], loc=mean, scale=std)
    print(sum(y))
    ax.plot(x, y, label="Exp {0}".format(i+1))
    ax.set_xlabel("$E^0(V)$")
    ax.set_ylabel("p($E^0$)")
    #ax.yaxis.set_tick_params(rotation=90)
fig.set_size_inches(3.5, 4.5)
ax.legend(loc="upper right")
plt.subplots_adjust(top=0.967,
bottom=0.115,
left=0.251,
right=0.957,
hspace=0.2,
wspace=0.2)
plt.show()
fig.savefig("recovered_E0.png", dpi=500)
