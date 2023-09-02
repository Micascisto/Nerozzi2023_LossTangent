# IMPORT LIBRARIES ETC.

import pandas as pd
pd.set_option("max_colwidth", None)
pd.set_option("display.max_rows", None)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy import stats

#----------------------------------------------------------------

# LOAD SHARAD CALIBRATED CHIRP SPECTRUM

# Import SHARAD chirp power spectrum file
chirp_spec = pd.read_csv('SHARAD_power_spectrum.csv')

#----------------------------------------------------------------

# LOAD RADAR DATA

# Region name if null
region_name = "Region_Name"

# Import csv radar data
data = pd.read_csv('Region_Name.csv')

# Go through each profile and remove picks with low SNR (using 2-σ from local background noise)
profile_list = data['profile'].unique().tolist()
for p in profile_list:
    subset = data[data['profile']==p]
    noise_mean = np.mean(subset['noise_p'])
    noise_std = np.std(subset['noise_p'])
    data = data.drop( data[ (data['profile']==p) & (data['sub_p']<(noise_mean + noise_std*2)) ].index)

# Add Region column if it doesn't exist
if 'Region' not in data:
    data['Region'] = region_name
    
# Adjust time to μs and calculate time delay and power loss
data['surface_t'] = data['surface_t']
data['sub_t'] = data['sub_t']
data['dt'] = data['sub_t'] - data['surface_t']
data['dp'] = data['sub_p'] - data['surface_p']

# Remove shallow returns too close to surface (<10 samples) and outliers (> 5-σ sigma)
data = data[(data['dt'] > 0.375e-6)]
data = data[(data['dp'] < 0)]

# Perform all tanδ calculations region by region
region_list = data['Region'].unique().tolist()
results = pd.DataFrame()
for r in region_list:
    data_sub = data[data['Region']==r]
     
    # Fit power loss vs time delay and calculate fit error
    p, V = np.polyfit(data_sub['dt'], data_sub['dp'], 1, cov=True)
    L_dB = p[0]
    L_dB_err = np.sqrt(V[0][0])
    intercept = p[1]
    intercept_err = np.sqrt(V[1][1])
    
    # Calculate tanδ
    L = 10**(L_dB / 10)
    c = 299792458
    wl = c / chirp_spec['frequency']
    k = wl / (2 * np.pi * c) * 1e6
    tan_d_f = -k * np.log(L)
    tan_d = np.average(tan_d_f, weights=chirp_spec['normalized_amplitude'])
    
    # Calculate error of tanδ
    tan_d_err_f = L_dB_err * np.log(10)/10 * k
    tan_d_err = np.average(tan_d_err_f, weights=chirp_spec['normalized_amplitude'])
    
    # Save everything in results df
    tmp = pd.DataFrame({'Region': [r], 'Loss_tangent': [tan_d], 'Error': [tan_d_err]})
    results = pd.concat([results, tmp], ignore_index = True)

    # Set up plots
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    plt.title("" + r, fontsize=18)
    plt.xlabel("Time delay (µs)", fontsize=16)
    plt.ylabel("Power loss (dB)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Power loss vs time delay
    plt.scatter(data_sub['dt'], data_sub['dp'])

    # Fit
    plt.plot(data_sub['dt'], L_dB * data_sub['dt'] + intercept, color='red')
    
    # Text box
    textstr = '\n'.join((
    r"dB/µs = %.3f ± %.3f" % (-L_dB, L_dB_err),
    r"tanδ = %.4f ± %.4f" % (tan_d, tan_d_err),
    ))
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)

# Save results into a csv file
results.to_csv("tan_d_results_Campbell.csv", mode='a', index=False)