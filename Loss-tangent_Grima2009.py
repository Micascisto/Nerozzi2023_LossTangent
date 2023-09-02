# IMPORT LIBRARIES ETC.

import pandas as pd
pd.set_option("max_colwidth", None)
pd.set_option("display.max_rows", None)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import scipy

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

# Go through each profile and remove picks with low SNR (using 2-σ from each profile background noise)
profile_list = data['profile'].unique().tolist()
for p in profile_list:
    subset = data[data['profile']==p]
    noise_mean = np.mean(subset['noise_p'])
    noise_std = np.std(subset['noise_p'])
    data = data.drop( data[ (data['profile']==p) & (data['sub_p']<(noise_mean + noise_std*2)) ].index )

# Add region name if it doesn't exist
if 'Region' not in data:
    data['Region'] = region_name

# Adjust time to s (from µs) and calculate time delay
data['surface_t'] = data['surface_t'] * 1e-6
data['sub_t'] = data['sub_t'] * 1e-6
data['dt'] = (data['sub_t'] - data['surface_t'])


# LOSS TANGENT CALCULATION AND PLOTTING

# Surface e' estimate
e_surf = 3.5

# Speed of light
c = 299792458

# Perform all tanδ calculations region by region
region_list = data['Region'].unique().tolist()
for r in region_list:
    data_sub = data[data['Region']==r]    
    results = data_sub

    for index, trace in data_sub.iterrows():
        # Calculate Pt
        Pt = trace['surface_p'] * (( (np.sqrt(e_surf)+1) / (np.sqrt(e_surf)-1) )**2 -1)

        # Calculate k = 0.091*f*c, taking chirp power spectrum into account    
        k = 0.091 * chirp_spec['frequency']/1e6 * c

        # Calculate tanδ
        tan_d_f = 10*np.log10(Pt/trace['sub_p']) / (k * trace['dt'])
        tan_d = np.average(tan_d_f, weights=chirp_spec['normalized_amplitude'])
    
        results.loc[index, 'tan_d'] = tan_d

    # Remove shallow returns too close to surface (<10 samples), negative unphysical loss tangents, and outliers (> 5-σ sigma)
    results = results[(results['dt'] > 0.375e-6)]
    results = results[(results['tan_d'] > 0)]
    results = results[((scipy.stats.zscore(results['tan_d'])) < 5)]

    # Calculate average and standard deviation of tanδ
    tand_avg = np.average(results['tan_d'])
    tand_std = np.std(results['tan_d'])
        
    # Set up plots
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    plt.xlabel("Loss tangent (tanδ)", fontsize=16)
    plt.ylabel("", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Histogram
    results.hist(column='tan_d', ax=ax, bins=50)
    plt.title(r)
    
    # Text box
    textstr = '\n'.join((
    r"surface ε' = %.1f" % (e_surf),
    r"tanδ = %.4f ± %.4f" % (tand_avg, tand_std),
    ))
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    
    plt.show()
    
    # Scatter plot of tanδ vs TWT    
    plt.scatter(results['dt']*1e6, results['tan_d'])
    plt.xlabel("TWT (µs)", fontsize=16)
    plt.ylabel("Loss tangent (tanδ)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

# Save results into a csv file
data.to_csv("tan_d_results_Grima.csv", mode='a', index=False)