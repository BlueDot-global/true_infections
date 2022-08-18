# Import packages
import os
from functools import total_ordering
from socket import AF_ATMSVC
import pandas as pd
import random
import math
import seaborn as sns
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from numpy.random import Generator
from datetime import datetime, timedelta
from collections import Counter
import sys
import datetime
import timeit
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from datetime import date
import csv

##os.chdir("/Users/laurenmckenzie/PycharmProjects/Underestimation of COVID Japan")

# Data Filename
FNAME = "data/DataJapan.csv"  # if using Canada data, have to adjust some code for column headers as they are not standardized

# Set seeds
random.seed(2442)
np.random.seed(2442)  # WHY in the world can't python just use a single RNG object

# Define IFR for each age group. Starts at age = 1, ends with age = 100.
ifr_lancet_original = [0.0054, 0.004, 0.0032, 0.0027, 0.0024, 0.0023, 0.0023, 0.0023, 0.0025, 0.0028,
                       0.0031, 0.0036, 0.0042, 0.005, 0.006, 0.0071, 0.0085, 0.01, 0.0118, 0.0138,
                       0.0162, 0.0188, 0.0219, 0.0254, 0.0293, 0.0337, 0.0386, 0.0442, 0.0504, 0.0573,
                       0.065, 0.0735, 0.0829, 0.0932, 0.1046, 0.1171, 0.1307, 0.1455, 0.1616, 0.1789,
                       0.1976, 0.2177, 0.2391, 0.262, 0.2863, 0.3119, 0.3389, 0.3672, 0.3968, 0.4278,
                       0.4606, 0.4958, 0.5342, 0.5766, 0.6242, 0.6785, 0.7413, 0.8149, 0.9022, 1.0035,
                       1.1162, 1.2413, 1.3803, 1.5346, 1.7058, 1.8957, 2.1064, 2.3399, 2.5986, 2.8851,
                       3.2022, 3.5527, 3.9402, 4.3679, 4.8397, 5.3597, 5.932, 6.5612, 7.252, 8.0093,
                       8.8381, 9.7437, 10.7311, 11.8054, 12.9717, 14.2346, 15.5984, 17.0669, 18.6431, 20.3292,
                       22.1263, 24.0344, 26.0519, 28.176, 30.4021, 32.7239, 35.1335, 37.6213, 40.1762, 42.7856]

# this picks the "4th" value for each age group (i.e. the median) 
# so, [4, 14, 24, ..., 94] (need the 95 there since it's an open end)
ifr_lancet_medians = [ifr_lancet_original[x] for x in np.arange(4, 95, 10) for i in range(10)]

# Comment this out to use the middle of the subset of list of Lancet IFR values; otherwise the original
# .. `ifr_lancet_original` will be used
ifr_lancet = np.array(ifr_lancet_original)

# Distribution parameter values. # WARNING: confirm if these values remain the same
ttd_param1 = 0.385  # original value: 0.385
ttd_param2 = 1 / 0.21  # original value: 1 / 0.21
inc_param1 = 0.661  # original value: 0.661
inc_param2 = 0.3604  # original value: 0.3604
sd = 1.35  # original value: 1


def pandas_output_setting():
    """Set pandas output display setting"""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 170)
    pd.options.mode.chained_assignment = None  # default='warn'


def prop_deaths_by_age() -> list:
    """
    Calculate the proportion of deaths for each age
    Data Sources
        - (for cumulative number of deaths) https://dc-covid.site.ined.fr/en/data/japan/
        - (population census data) https://www.e-stat.go.jp/en/stat-search/files?page=1&layout=datalist&toukei=00200521&tstat=000001136464&cycle=0&year=20200&month=24101210&tclass1=000001136466
        - (Infection Fatality Ratio - Table 1) https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)02867-1/fulltext#seccestitle140
        - (population estimates -- not used) https://www.stat.go.jp/english/data/jinsui/tsuki/index.html
        - (population pyramid -- not used) https://worldpopulationreview.com/countries/japan-population
    :return: a list that contains the proportion of deaths by age, from age = 1 to age = 100
    """
    # This is the size of the population per age. Starts at age = 1, and ends at age = 100
    pop_sizes = np.array(
        [866525, 910005, 934063, 973665, 998664, 996576, 1020657, 1024325, 1048871, 1057736, 1063185, 1083113, 1077379,
         1069104, 1070370, 1113159, 1123237, 1151389, 1159285, 1177049, 1174456, 1193935, 1193983, 1191883, 1210380,
         1212167, 1192761, 1209983, 1206673, 1234225, 1258775, 1299569, 1335672, 1356353, 1407719, 1456774, 1476522,
         1478920, 1491632, 1558309, 1596338, 1656969, 1699648, 1779813, 1852394, 1956602, 1991137, 1954205, 1895955,
         1837431, 1807894, 1764215, 1758955, 1371356, 1689290, 1582910, 1542017, 1491947, 1461318, 1469672, 1494495,
         1450282, 1407740, 1475001, 1516441, 1512118, 1598839, 1679855, 1768015, 1886160, 2052521, 2014143, 1893386,
         1165585, 1234832, 1488292, 1416571, 1430462, 1360771, 1206058, 1017635, 1048747, 1035057, 989231, 893754,
         794162, 739149, 662252, 580506, 493824, 425042, 358386, 277613, 224151, 169988, 123057, 90075, 64897, 44707,
         32032])

    # This is the aggregated proportion of death. Age group aggregation as follows:
    # a)1-9, b)10-19, c)20-29, d)30-39, e)40-49, f)50-59, g)60-69, h)70-79, i)80+ # WARNING: may need to be replaced with more updated death rates
    aggregated_proportion_of_deaths = np.array(
        [0.0188033545184461, 0.0376067090368922, 0.2482042796434884, 0.5565792937460043, 1.2936707908690910,
         3.7644315745929074, 9.6762062351923583, 20.2963408672107104, 64.1081568951901019])

    # For testing
    '''aggregated_proportion_of_deaths = np.array(
        [11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1, 11.1])'''

    aggregated_proportion_of_deaths = [prop / 100 for prop in aggregated_proportion_of_deaths]

    # Distribute the aggregated proportion of deaths to each age based on the population size in the corresponding
    # .. age group.
    proportion_deaths_by_age = []

    # For age 1 - 9
    pop_sizes_1_to_9 = pop_sizes[0: 9]
    aggregated_proportion_of_deaths_1_to_9 = aggregated_proportion_of_deaths[0]
    proportion_deaths_1_to_9 = [i / sum(pop_sizes_1_to_9) * aggregated_proportion_of_deaths_1_to_9 for i in
                                pop_sizes_1_to_9]

    # For age 10 - 19
    pop_sizes_10_to_19 = pop_sizes[9: 19]
    aggregated_proportion_of_deaths_10_to_19 = aggregated_proportion_of_deaths[1]
    proportion_deaths_10_to_19 = [i / sum(pop_sizes_10_to_19) * aggregated_proportion_of_deaths_10_to_19 for i in
                                  pop_sizes_10_to_19]

    # For age 20 - 29
    pop_sizes_20_to_29 = pop_sizes[19: 29]
    aggregated_proportion_of_deaths_20_to_29 = aggregated_proportion_of_deaths[2]
    proportion_deaths_20_to_29 = [i / sum(pop_sizes_20_to_29) * aggregated_proportion_of_deaths_20_to_29 for i in
                                  pop_sizes_20_to_29]

    # For age 30 - 39
    pop_sizes_30_to_39 = pop_sizes[29: 39]
    aggregated_proportion_of_deaths_30_to_39 = aggregated_proportion_of_deaths[3]
    proportion_deaths_30_to_39 = [i / sum(pop_sizes_30_to_39) * aggregated_proportion_of_deaths_30_to_39 for i in
                                  pop_sizes_30_to_39]

    # For age 40 - 49
    pop_sizes_40_to_49 = pop_sizes[39: 49]
    aggregated_proportion_of_deaths_40_to_49 = aggregated_proportion_of_deaths[4]
    proportion_deaths_40_to_49 = [i / sum(pop_sizes_40_to_49) * aggregated_proportion_of_deaths_40_to_49 for i in
                                  pop_sizes_40_to_49]

    # For age 50 - 59
    pop_sizes_50_to_59 = pop_sizes[49: 59]
    aggregated_proportion_of_deaths_50_to_59 = aggregated_proportion_of_deaths[5]
    proportion_deaths_50_to_59 = [i / sum(pop_sizes_50_to_59) * aggregated_proportion_of_deaths_50_to_59 for i in
                                  pop_sizes_50_to_59]

    # For age 60 - 69
    pop_sizes_60_to_69 = pop_sizes[59: 69]
    aggregated_proportion_of_deaths_60_to_69 = aggregated_proportion_of_deaths[6]
    proportion_deaths_60_to_69 = [i / sum(pop_sizes_60_to_69) * aggregated_proportion_of_deaths_60_to_69 for i in
                                  pop_sizes_60_to_69]

    # For age 70 - 79
    pop_sizes_70_to_79 = pop_sizes[69: 79]
    aggregated_proportion_of_deaths_70_to_79 = aggregated_proportion_of_deaths[7]
    proportion_deaths_70_to_79 = [i / sum(pop_sizes_70_to_79) * aggregated_proportion_of_deaths_70_to_79 for i in
                                  pop_sizes_70_to_79]

    # For age 80 - 100
    pop_sizes_80_to_100 = pop_sizes[79:]
    aggregated_proportion_of_deaths_80_to_100 = aggregated_proportion_of_deaths[8]
    proportion_deaths_80_to_100 = [i / sum(pop_sizes_80_to_100) * aggregated_proportion_of_deaths_80_to_100 for i in
                                   pop_sizes_80_to_100]

    # Add to the final list
    add_lists = [proportion_deaths_1_to_9, proportion_deaths_10_to_19, proportion_deaths_20_to_29,
                 proportion_deaths_30_to_39, proportion_deaths_40_to_49, proportion_deaths_50_to_59,
                 proportion_deaths_60_to_69, proportion_deaths_70_to_79, proportion_deaths_80_to_100]

    for add_list in add_lists:
        for i in add_list:
            proportion_deaths_by_age.append(i)

    return np.array(proportion_deaths_by_age)


def read_death_data():
    """Read .csv file into df"""
    # Import data frame
    df = pd.read_csv(f'{FNAME}', parse_dates=['date'])
    df = df.sort_values(by="date")

    # create data frame by date
    df = df.groupby(['date'])[['dailyDeaths']].sum()
    df.reset_index(level=0, inplace=True)
    return df


def read_case_data():
    """Read .csv file into df"""
    # Import data frame
    df = pd.read_csv(f'{FNAME}', parse_dates=['date'])
    df = df.sort_values(by="date")
    return df


def distr_lag(input_array):
    """Redistribute number into earlier time (lag)"""
    input_array = input_array.astype(int)
    ttd = np.random.gamma(ttd_param1, ttd_param2, sum(input_array))
    inc = np.random.lognormal(inc_param1, inc_param2, sum(input_array))
    lags = (ttd + inc).astype(int)
    maxlag = max(lags)
    current_counter = np.zeros(
        len(input_array) + maxlag)  # create a number counter of all zeros that is the same length as total number of days (plus maximum lag time for model stability)
    lagptr = 0  # pointer to where we are in the lags
    for (i, x) in enumerate(input_array):
        _idx = range(lagptr, x + lagptr)
        inflags = lags[_idx]
        shftlags = maxlag + i - inflags  # shift the lag times to account for the extra elements in `current_counter`
        lagptr += x  # update the lag pointer
        np.add.at(current_counter, shftlags, 1)
    return current_counter[maxlag:(len(current_counter))]


def plot_redistributed_data(curr_df):
    """Show reported deaths and redistributed deaths with lags."""
    sns.lineplot(x='date', y='dailyDeaths', data=curr_df, label='Daily Deaths')
    sns.lineplot(x='date', y='deaths_redistributed', data=curr_df, label='Redistributed Deaths')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel("Daily Deaths")
    plt.show()


def split_redistributed_deaths_by_age(curr_df, prop_deaths_per_age):
    """Split the daily death by age group using `prop_deaths_per_age`."""
    for (i, row) in enumerate(prop_deaths_per_age):
        # print(i, row)
        varname = 'output_age_group{}'.format(i + 1)
        curr_df[varname] = curr_df['deaths_redistributed'] * row

    curr_df['_death_sum'] = curr_df.loc[:, curr_df.columns.str.contains('output_age_group')].sum(axis=1)
    return curr_df

def calculate_infected_cases_in_single_sim(curr_df, percent_reduce):
    """Go through each day's death-output columns (`output_age_group1`-`output_age_group100`), apply age-specific IFR to
    obtain inferred age-specific infected cases, then add all the inferred infected cases into a single column.
    The percent_reduce specifies the reduction in IFR values (and is the same length as total time)"""
    mu = np.log(ifr_lancet)  # shape (100,)
    nba = curr_df[curr_df.columns[curr_df.columns.str.contains("output_age_group")]].values.tolist()  # len 729
    ndays = len(nba)
    totalvalues = np.zeros((ndays, 100))  # shape (729, 100)

    # calculate the infected cases via dividing age-specific deaths by age-specific IFRs
    for (i, row) in enumerate(nba):
        mu = np.log(ifr_lancet * percent_reduce[i]) # reduce IFR value 
        s1 = np.random.lognormal(mu, sd, (1, len(mu))) / 100  # sample from IFR distributions
        inf = (row / s1).astype(int)  # divide by IFR to get cases at the per age
        totalvalues[i, :] = inf
    return totalvalues

def plot_infections(dailycases, ti_mean, ti_min, ti_max, cumulative = False):
    """Plots the infections plus the uncertainty bands"""

    # Initiate dates for x-axis
    x = np.random.randint(low=0, high=50, size=729)
    plt.figure()
    plt.xticks(np.arange(0, len(x) + 1, 175))
    start = datetime.date(2020, 1, 1)
    oneday = datetime.timedelta(1)
    labels = [date.isoformat() for date in (start + oneday * i for i in range(0, 729))] 
    fname = "InfectionsByDate.pdf"

    if cumulative: 
        ti_mean = np.cumsum(ti_mean)
        ti_min = np.cumsum(ti_min) 
        ti_max = np.cumsum(ti_max) 
        dailycases = np.cumsum(dailycases)
        # june 15 2020  tidx = 151 
        # december 15 2020 tidx = 334 
        # december 27 2020 tidx = 711
        psz = 128000000
        print(f'first sero (june 15 tidx) {ti_mean[165] / psz} ({ti_min[165]/psz} - {ti_max[165]/psz})')
        print(f'sec sero (dec 15 tidx) {ti_mean[333] / psz} ({ti_min[333]/psz} - {ti_max[333]/psz})')
        print(f'third sero (dec 27 tidx) {ti_mean[710] / psz} ({ti_min[710]/psz} - {ti_max[710]/psz})') 
        fname = "CumulativeInfectionsByDate.pdf"

    # Plot data
    plt.plot(labels, ti_mean / 1000000, label='Lag-adjusted estimated cases', color="blue")
    plt.fill_between(np.arange(len(ti_mean)), ti_min / 1000000, ti_max / 1000000, color="#7f7fff", alpha=0.2)
    plt.plot(labels, dailycases / 1000000, color='#7a5d00', label="Reported infections", alpha=0.4)

    # Plot meta-analysis results on the cumulative plot
    if cumulative:
        # random effect
        #plt.plot(labels[160], 151771.7532 / 1000000, 'o', markersize=3, color='black')
        #plt.vlines(labels[160], 63238.2305 / 1000000, 316191.1525 / 1000000, color='black')
        #plt.plot(labels[348], 733563.4738 / 1000000, 'o', markersize=3, color='black')
        #plt.vlines(labels[348], 404724.6752 / 1000000, 1328002.8405 / 1000000, color='black')
        plt.plot(labels[160], 151771.7532 / 1000000, 'o', markersize=3, color='black')
        plt.vlines(labels[160], 63238.2305 / 1000000, 303543.1525 / 1000000, color='black')
        plt.plot(labels[348], 1049754.4738 / 1000000, 'o', markersize=3, color='black')
        plt.vlines(labels[348], 860039 / 1000000, 1264764 / 1000000, color='black')
        plt.plot(labels[707], 3000009 / 1000000, 'o', markersize=3, color='black')
        plt.vlines(labels[707], 2596647 / 1000000, 3479002 / 1000000, color='black')

    # Add labels and a legend
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.1)
    plt.xlabel("Date")
    plt.ylabel("Cumulative infections (in millions)")
    blue_line = mlines.Line2D([], [], color='blue', linewidth=0.9, label='Estimated')
    red_patch = mlines.Line2D([], [], color='#7a5d00', alpha=0.4, linewidth=0.9, label='Reported')
    grey_line = mlines.Line2D([], [], color='black', marker='o', linewidth=0.9, markersize=4,
                             label='Seroprevalence estimate')
    plt.legend(handles=[red_patch, blue_line, grey_line], frameon=False, loc="upper left")
    plt.savefig(f'output/{fname}', format="pdf")
    plt.show()

def calculate_ifr_reductions(tidx):
    # Calculate IFR reduction values. The reduction is obtained from Fig 3 https://www.nature.com/articles/s41598-021-03269-w
    # where we calculate the percent decrease from the average IFR in 2020 (pre-vaccine, blue dotted line) 
    # and the AVERAGE of the IFR from 2021/01/01 to the end of the figure.
    # the average value before 2021/01 is roughly 0.007426 and after 2021/01 is 0.00312302, roughly a 58% decrease
    # we construct a linear curve so that 2021/01 is 0% decrease and by the end is a 50% decrease
    # (the extraction of the data of Figure 3 from a digitizer is in nature_ifr_values.csv)
    # (another way was to get to 86.5% exponentially, but this produces very large, unrealistic results)
    print('Calculating IFR reduction vector')
    pr = np.zeros(tidx)
    
    # y = 0.000159117 * np.power(1.02523, range(351, 530)) / 100  # exponential curve y = ab^x
    #y = 0.22619 * np.arange(351, 729) - 78.393 # linear curve y = mx + b
    y = 0.13228 * np.arange(351, 729) - 46.429 # 55% decrease from january 1, 2021 to 
    y = 0.15106 * np.arange(398, 729) - 60.121 # 50% decrease from feb 17 to december 31
    y = 0.16616 * np.arange(398, 729) - 66.133 # 55% decrease from feb 17 to december 31
    y = y / 100
    pr[398:729] =  y 
    plt.figure() 
    plt.plot(pr, color="blue")
    plt.xlabel('Time Unit')
    plt.ylabel("IFR multiplicative factor")
    plt.savefig("output/ifr_reduction_percent.pdf", format = 'pdf')
    return(1 - pr)


def main():
    # Get death proportion by age
    prop_deaths_per_age = prop_deaths_by_age()

    # Set simulation parameters
    print('Starting main program')
    death_redistribution_method = 'per_sim'  # outputs can be 'presampled' or 'per_sim'
    sim_n_repeats = 1000
    death_redistribution_n_repeats = 1000
    start = timeit.default_timer()
    pandas_output_setting()

    # Load in death and case data
    print('Reading csv files')
    df_death = read_death_data()
    death_array = df_death.loc[:, 'dailyDeaths']
    df_case = read_case_data()
    case_array = df_case.dailyCases 

    # Initialize placeholders
    death_redistributed_array_list = []
    ti = []
    ag = []
    ag_means = []
    ag_mins = []
    ag_maxes = []

    # Calculate IFR reduction values
    pr = calculate_ifr_reductions(len(death_array))

    # Sample death lags before simulations start (if option is set)
    if death_redistribution_method == 'presampled':
        for _ in range(death_redistribution_n_repeats):
            death_redistributed = distr_lag(death_array)
            death_redistributed_array_list.append(death_redistributed)

    # Run multiple sims
    for _ in range(sim_n_repeats):
        if death_redistribution_method == 'presampled':
            death_redistributed = random.choice(death_redistributed_array_list)

        elif death_redistribution_method == 'per_sim':
            death_redistributed = distr_lag(death_array)

        else:
            assert False, 'Enter either `presampled` or `per_sim` for `death_redistribution_method`.'

        df_death['deaths_redistributed'] = death_redistributed
        df_death = split_redistributed_deaths_by_age(curr_df=df_death, prop_deaths_per_age=prop_deaths_per_age)
        _tv = calculate_infected_cases_in_single_sim(curr_df=df_death, percent_reduce=pr)
        ti.append(np.sum(_tv, axis=1)) # sums up across age groups to get 729 numbers for total infections
        ag.append(np.sum(_tv, axis=0)) # sums each age across time to get 100 different numbers for each age

    ti = np.asarray(ti, dtype=np.float32)  # ti is nsims x ntime
    ag = np.asarray(ag, dtype=np.float32)  # ag is nsims x ages 
    print(f'total time x sims {ti.shape}')

    # Write the raw output files 
    with open("output/simulation_output_totalinfections_time.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(ti)
    with open("output/simulation_output_totalinfections_ages.csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(ag)

    # Calculate the number of infections over time
    ti_mean = np.mean(ti, axis=0)
    ti_min = np.quantile(ti, 0.10, axis=0)
    ti_max = np.quantile(ti, 0.90, axis=0)

    ag_mean_sp = np.mean(ag, axis=0)
    ag_min_sp = np.quantile(ag, 0.10, axis=0)
    ag_max_sp = np.quantile(ag, 0.90, axis=0)

    # Print statistics 
    rep = int(sum(case_array))
    sim = int(sum(ti_mean))
    slo = int(sum(ti_min))
    shi = int(sum(ti_max))
    print(f'reported sum of infections: {rep}')
    print(f'rep inf: {rep} sim inf: {sim} ({slo} - {shi})')
    print(f'x higher: {sim / rep}')
    print(f'case ascertainment: {rep / sim} ({rep / slo} - {rep / shi})')
    print(f'pop japan %: {sim / 128000000}')
    # Sum age-groups every 10 groups 
    # so 1 - 10, 11 - 20, 21 - 30, etc etc. 
    tf_age_mean = np.reshape(ag_mean_sp, (-1, 10)).sum(axis = 1)
    tf_age_min = np.reshape(ag_min_sp, (-1, 10)).sum(axis = 1)
    tf_age_max = np.reshape(ag_max_sp, (-1, 10)).sum(axis = 1)
    
    # plot infections by date
    plot_infections(df_case['dailyCases'], ti_mean, ti_min, ti_max, False)
    plot_infections(df_case['dailyCases'], ti_mean, ti_min, ti_max, True)

    # Plot number of infections by age:
    plt.figure()
    plt.style.use('default')
    # Plot data
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xlab = [str(i) for i in x]
    xlab = ["1 - 10", "11 - 20", "21 - 30", "31 - 40", "41 - 50", "51 - 60", "61 - 70", "71 - 80", "81 - 90", "91 - 100"]
    yerror = np.array([tf_age_mean - tf_age_min, tf_age_max - tf_age_mean])
    plt.errorbar(x, tf_age_mean, yerr=yerror, fmt='o', color="mediumblue", capsize=4, label='Estimated', markersize=4,
                 linewidth=0.8)

    rep_infections_age = [94046, 174469,424964,281203,265446,210861,104272,78502, 53922, 20204]
    # Add reported number of cases
    plt.vlines(x[0], 0, rep_infections_age[0], color='silver', alpha=0.5, linewidth=5.0, label='Reported')
    plt.vlines(x[1], 0, rep_infections_age[1], color='silver', alpha=0.5, linewidth=5.0)
    plt.vlines(x[2], 0, rep_infections_age[2], color='silver', alpha=0.5, linewidth=5.0)
    plt.vlines(x[3], 0, rep_infections_age[3], color='silver', alpha=0.5, linewidth=5.0)
    plt.vlines(x[4], 0, rep_infections_age[4], color='silver', alpha=0.5, linewidth=5.0)
    plt.vlines(x[5], 0, rep_infections_age[5], color='silver', alpha=0.5, linewidth=5.0)
    plt.vlines(x[6], 0, rep_infections_age[6], color='silver', alpha=0.5, linewidth=5.0)
    plt.vlines(x[7], 0, rep_infections_age[7], color='silver', alpha=0.5, linewidth=5.0)
    plt.vlines(x[8], 0, rep_infections_age[8], color='silver', alpha=0.5, linewidth=5.0)
    plt.vlines(x[9], 0, rep_infections_age[9], color='silver', alpha=0.5, linewidth=5.0)

    print(f'age specific infections: {tf_age_mean}')
    print(f'age specific lo: {tf_age_min}')
    print(f'age specific hi: {tf_age_max}')
    print(f'age underascertainment {rep_infections_age / tf_age_mean}')
    
    # Add labels and a legend
    blue_line = mlines.Line2D([], [], color='mediumblue', marker='o', linewidth=0.9, markersize=4,
                              label='Estimated infections')
    grey_line = mlines.Line2D([], [], color='silver', alpha=0.5, linewidth=4, label='Reported infections')
    plt.legend(handles=[grey_line, blue_line], frameon=False, fontsize=9)
    plt.xticks(x, xlab)
    plt.ylabel('Cumulative infections')
    plt.xlabel("Age group")
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(top=0.97)
    plt.subplots_adjust(top=0.97)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in plt.gca().get_yticks()])
    plt.savefig('output/cumulativeInfectionsByAge.pdf', format="pdf")
    plt.show()


if __name__ == '__main__':
    main()
