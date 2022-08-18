# This file is now deprecated. 
# Use `main.py` instead

# This program estimates the true number of COVID-19 infections given IFR values and total number of deaths
from functools import total_ordering
import pandas as pd
import math
import seaborn as sns
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time
from numpy.random import Generator
from datetime import datetime, timedelta
from collections import Counter

# Define some constants.
# Proportion of total deaths attributed to each age group
# Age groups for analysis: 0-9, 10-19, 2:20-29, 3:30-39, 4:40-49, 5:50-59, 6:60-69, 7:70-79, 8:80+
# As of July 23rd, Source: https://health-infobase.canada.ca/covid-19/epidemiological-summary-covid-19-cases.html
propdeaths_byage = np.array([0.0188033545184461, 0.0376067090368922, 0.2482042796434884, 0.5565792937460043, 1.2936707908690910, 3.7644315745929074, 9.6762062351923583, 20.2963408672107104, 64.1081568951901019])
propdeaths_byage = propdeaths_byage / 100 

# Define IFR for each age group, we do not use Verity any more... only Driscoll
#IFR_verity = [0.00161, 0.0309, 0.0844, 0.161, 0.595, 1.93, 4.28, 7.80]
IFR_driscoll = [0.002, 0.003, 0.013, 0.04, 0.12, 0.32, 1.07, 3.2, 8.29]

def read_data(): 
    # Import data frame
    cov = pd.read_csv("COVIDdd.csv", parse_dates=['date'])
    # Print last day included in data frame : currently updated to August 3rd 2021
    print(cov["date"].iloc[-1])

    #create non-cumulative deaths by day
    cov['lag'] = cov["numdeaths"].shift(1).fillna(0)
    cov['deaths_byday'] = cov["numdeaths"] - cov.lag

    #create non-cumulative infections by day
    cov['lag2'] = cov["numconf"].shift(1).fillna(0)
    cov['cases_byday'] = cov["numconf"] - cov.lag2
    cov['7day_rolling_avg'] = cov["deaths_byday"].rolling(7).mean()
    return cov 

def plot_data():
    cov = read_data()
    sns.lineplot(x='date', y='deaths_byday', data=cov, label='Daily Deaths')
    sns.lineplot(x = 'date', y = '7day_rolling_avg', data = cov, label = 'Rolling 7 Average')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel("Daily Deaths")
    plt.show()
    # Plot cumulative deaths over time
    sns.lineplot(x='date', y='numdeaths', data=cov, label='Daily Deaths')
    plt.xlabel('Date')
    plt.ylabel("Cumulative Deaths")
    plt.show()

def distr_lag(inf_array):
    inf_array = inf_array.astype(int)
    ttd = np.random.gamma(3.385, 1/0.21, sum(inf_array))  # time to death
    inc = np.random.lognormal(0.661, 0.3604, sum(inf_array)) # incubation
    lags = (ttd + inc).astype(int) 
    maxlag = max(lags) 
    infctr = np.zeros(len(inf_array) + maxlag) # create an infection counter of all zeros that is the same length as total number of days (plus maximum lag time for model stability)
    lagptr = 0 # pointer to where we are in the lags 
    for (i, x) in enumerate(inf_array):
        _idx = range(lagptr, x + lagptr) 
        inflags = lags[_idx] 
        shftlags = maxlag + i - inflags # shift the lag times to account for the extra elements in infctr 
        lagptr += x  # update the lag pointer
        np.add.at(infctr, shftlags, 1)
    return infctr[maxlag:(len(infctr))]

# Define function to infer total number of infections given IFR and total number of deaths 
def split_infer_add(deaths):
    mu = np.log(IFR_driscoll)
    nba = [propdeaths_byage * d for d in deaths]  
    ndays = len(nba)
    
    # stores 1000 samples of the added total number of infections, loses age-group accuracy
    _totalinfs = np.zeros((1000, ndays))  
    totalinfs = np.zeros((1000, ndays))
    # stores the mean, min, maxes of the 1000 simulations (i.e. stores everything at the age-group level)
    ag_means = np.zeros((ndays, 9))
    ag_mins = np.zeros((ndays, 9))
    ag_maxes = np.zeros((ndays, 9))
    # go through each day's deaths, apply IFR to estimate infections
    for (i, row) in enumerate(nba):
        s1 = np.random.lognormal(mu, 0.07, (1000, len(mu))) / 100  # sample from IFR distributions
        inf = (row / s1).astype(int)  # divide by IFR to get cases at the age group level
        _totalinfs[:, i] = np.sum(inf, axis=1)
        ag_means[i, :] = np.mean(inf, axis=0)
        ag_mins[i, :] = np.min(inf, axis=0)
        ag_maxes[i, :] = np.max(inf, axis=0)
    # for each of the 1000 MC simulations of total infections, run our lag algorithm to distribute the cases
    for (i, row) in enumerate(totalinfs):
        print(i)
        totalinfs[i, :] = distr_lag(_totalinfs[i, :])
    return (totalinfs, ag_means, ag_mins, ag_maxes)

cov = read_data()
deaths = np.array(cov.iloc[:, 7])
ti, ag_means, ag_mins, ag_maxes = split_infer_add(deaths)

# sum up daily deaths at the age group level
tf_mean = np.sum(ag_means, axis = 0)
tf_min = np.sum(ag_mins, axis = 0)
tf_maxes = np.sum(ag_maxes, axis = 0)

#plot cumulative estimated cases by age
y2=[103008, 172557,273181,234008,208694,185552,114654,60200,71130]
x=[1,2,3,4,5,6,7,8,9]
yerror = np.array([tf_mean - tf_min, tf_maxes - tf_mean])
plt.errorbar(x, tf_mean, yerr=yerror, fmt='o',alpha=0.4)
xlab = ['0-9', '10-19', '20-29', '30-39','40-49','50-59','60-60','70-79','80+']
plt.plot(x, y2, 'm*', label='Reported Number of Cases')
plt.legend(loc='upper left', frameon=False)
plt.xticks(x, xlab)
plt.yticks(rotation=45)
plt.ylabel('Confirmed of Estimated Cases')
plt.xlabel("Age Group")
plt.show()

# plot the cumulative number of infections over time
ti_mean = np.cumsum(np.mean(ti, axis=0))
ti_min = np.cumsum(np.min(ti, axis=0))
ti_max = np.cumsum(np.max(ti, axis=0))
plt.plot(np.arange(len(ti_mean)), ti_mean, label='Lag-adjusted Cases')
plt.plot(np.arange(len(ti_mean)), ti_min, ti_max)
plt.legend(loc='upper left', frameon=False)
plt.show()