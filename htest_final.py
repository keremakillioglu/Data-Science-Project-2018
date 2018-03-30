import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statistics
from collections import Counter

growth_table = pd.read_csv('state_growth.csv')
champions_table = pd.read_csv('champions.csv')

years = list(champions_table['Year'])
years = years[:-1]

nfl_champions = list(champions_table['NFL Champion State'])
nfl_champions = nfl_champions[1:-1]
nfl_champions = list(set(nfl_champions))
nba_champions = list(champions_table['NBA Champion State'])
nba_champions = list(set(nba_champions))
mbl_champions = list(champions_table['MBL Champion State'])
mbl_champions = list(set(mbl_champions))

champions_list_with_year = champions_table.values.tolist()
champions_list_with_year = champions_list_with_year[::-1]

#football
temp_football_champions = nfl_champions
nfl_champ_with_years = []
for item in champions_list_with_year:
    if item[1] in temp_football_champions:
        d = []
        d.append(item[1])
        d.append(item[0])
        nfl_champ_with_years.append(d)
        temp_football_champions.remove(item[1])

before_championship_nfl = []
after_championship_nfl = []
for item in nfl_champ_with_years:
    row = growth_table.loc[growth_table['State'] == item[0]]
    np_row = row.as_matrix()
    growths = list(np_row[0])
    index = item[1] % 2000
    before = growths[1:index+1]
    after = growths[index+1:] 
    if len(after) != 0 and len(before) != 0:
        m1 = statistics.mean(before)
        m2 = statistics.mean(after)
        before_championship_nfl.append(m1)
        after_championship_nfl.append(m2)

stats.ttest_ind(before_championship_nfl, after_championship_nfl, axis = 0,nan_policy='propagate')
years = years[::-1]

#basketball
temp_basketball_champions = nba_champions
nba_champ_with_years = []

for item in champions_list_with_year:
    if item[2] in temp_basketball_champions:
        d = []
        d.append(item[2])
        d.append(item[0])
        nba_champ_with_years.append(d)
        temp_basketball_champions.remove(item[2])

before_championship_nba = []
after_championship_nba = []
for item in nba_champ_with_years:
    row = growth_table.loc[growth_table['State'] == item[0]]
    np_row = row.as_matrix()
    growths = list(np_row[0])
    index = item[1] % 2000
    before = growths[1:index+1]
    after = growths[index+1:] 
    if len(after) != 0 and len(before) != 0:
        m1 = statistics.mean(before)
        m2 = statistics.mean(after)
        before_championship_nba.append(m1)
        after_championship_nba.append(m2)

stats.ttest_ind(before_championship_nba, after_championship_nba, axis = 0,nan_policy='propagate')

#baseball
temp_baseball_champions = mbl_champions
mbl_champ_with_years = []

for item in champions_list_with_year:
    if item[3] in temp_baseball_champions:
        d = []
        d.append(item[3])
        d.append(item[0])
        mbl_champ_with_years.append(d)
        temp_baseball_champions.remove(item[3])

before_championship_mbl = []
after_championship_mbl = []
for item in mbl_champ_with_years:
    row = growth_table.loc[growth_table['State'] == item[0]]
    np_row = row.as_matrix()
    growths = list(np_row[0])
    index = item[1] % 2000
    before = growths[1:index+1]
    after = growths[index+1:] 
    if len(after) != 0 and len(before) != 0:
        m1 = statistics.mean(before)
        m2 = statistics.mean(after)
        before_championship_mbl.append(m1)
        after_championship_mbl.append(m2)

stats.ttest_ind(before_championship_mbl, after_championship_mbl, axis = 0,nan_policy='propagate')

nfl_plot_list = []
for champ in nfl_champ_with_years:
    g = growth_table.loc[growth_table['State'] == champ[0]]
    np_df = g.as_matrix()
    k = list(np_df[0])
    k = k[1:]
    state = [years, k, champ[0], champ[1]]
    nfl_plot_list.append(state)

fig = plt.figure(figsize=(12, 9))
plt.plot()
plt.xlabel('Years')
plt.ylabel('Growth Rate')
for state in nfl_plot_list:
    champion = state[3]
    t = 0
    for idx, year in enumerate(state[0]):
        
        if champion == year:
            t = idx
            break
    plt.plot(years, state[1], '-D', label=state[2], markevery=[t])
plt.legend(loc=(1.05,0.5))
plt.show()

nba_plot_list = []
for champ in nba_champ_with_years:
    g = growth_table.loc[growth_table['State'] == champ[0]]
    np_df = g.as_matrix()
    k = list(np_df[0])
    k = k[1:]
    state = [years, k, champ[0], champ[1]]
    nba_plot_list.append(state)

fig = plt.figure(figsize = (12, 9))
plt.plot()
plt.xlabel('Years')
plt.ylabel('Growth Rate')
for state in nba_plot_list:
    champion = state[3]
    t = 0
    for idx, year in enumerate(state[0]):
        if champion == year:
            t = idx
            break
    plt.plot(years, state[1], '-D', label = state[2], markevery = [t])
plt.legend(loc = (1.05, 0.81))
plt.show()

mbl_plot_list = []
for champ in mbl_champ_with_years:
    g = growth_table.loc[growth_table['State'] == champ[0]]
    np_df = g.as_matrix()
    k = list(np_df[0])
    k = k[1:]
    state = [years, k, champ[0], champ[1]]
    mbl_plot_list.append(state)

fig = plt.figure(figsize = (12, 9))
plt.plot()
plt.xlabel('Years')
plt.ylabel('Growth Rate')
for state in mbl_plot_list:
    champion = state[3]
    t = 0
    for idx, year in enumerate(state[0]):
        if champion == year:
            t = idx
            break
    plt.plot(years, state[1], '-D', label = state[2], markevery = [t])
plt.legend(loc = (1.05, 0.75))
plt.show()

# all states growth rate plot
years = list(champions_table['Year'])
years = years[1:]
years = years[::-1]
states = list(growth_table['State'])
growth_rates = []
for state in states:
    row = growth_table.loc[growth_table['State'] == state]
    np_row = row.as_matrix()
    growths = list(np_row[0])
    growths = growths[1:]
    growth_rates.append(growths)

all_data = []
for state, growth in zip(states, growth_rates):
    temp = [years, state, growth]
    all_data.append(temp)

fig = plt.figure(figsize=(16, 12))
plt.plot()
plt.xlabel('Years')
plt.ylabel('Growth Rate')
for state in all_data:
    plt.scatter(state[0], state[2], label=state[1])
plt.legend(loc=(1.05,0))
plt.show()

# nfl champion count bar graph
all_nfl_champions = []
col = champions_table['NFL Champion State']
for state in col:
    all_nfl_champions.append(state)

champ_count = Counter(all_nfl_champions)
fig = plt.figure(figsize = (12, 9))
plt.plot()
plt.xlabel('States')
plt.ylabel('Number of NFL Championships')
for key in champ_count:
    plt.bar(key, champ_count[key])
plt.show()

# nba champion count bar graph
all_nba_champions = []
col = champions_table['NBA Champion State']
for state in col:
    all_nba_champions.append(state)

champ_count = Counter(all_nba_champions)
fig = plt.figure(figsize = (12, 9))
plt.plot()
plt.xlabel('States')
plt.ylabel('Number of NBA Championships')
for key in champ_count:
    plt.bar(key, champ_count[key])
plt.show()

# mbl champion count bar graph
all_mbl_champions = []
col = champions_table['MBL Champion State']
for state in col:
    all_mbl_champions.append(state)

champ_count = Counter(all_mbl_champions)
fig = plt.figure(figsize = (12, 9))
plt.plot()
plt.xlabel('States')
plt.ylabel('Number of MBL Championships')
for key in champ_count:
    plt.bar(key, champ_count[key])
plt.show()
