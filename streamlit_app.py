#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mark
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import dirichlet

import streamlit as st

st.write("# How fair is your dreidel?")

intro = st.expander("Introduction")
body = st.expander("Data Input & Results")
methods = st.expander('Methodology')
misc = st.expander('Misc')

intro.write('''Dreidels are notoriously biased. This app helps you visualize how fair your dreidel is using bayesian statistics. 

All you have to do is spin your dreidel many times and record where it lands. Then input the total numbers into the "Data Input" section. A graph will show you the probability distribution of your dreidel.

You can even input the data as you go and watch the probabilities change over time.''')

body.write('Input the number of times your dreidel landed on each of the following sides:')

n_nun = body.number_input('Number of נ (Nun)', min_value=0)
n_gimel = body.number_input('Number of ג (Gimel)', min_value=0)
n_hei = body.number_input('Number of ה (Hei)', min_value=0)
n_shin = body.number_input('Number of ש (Shin)', min_value=0)


###
#Do the calculation
###

sns.set_context('talk', font_scale=.9)
n_sides = 4
prior_params = 5 * np.ones(n_sides)
resolution = 1000 # how many data points of the distribution to sample. 1000 gives a point every 0.1%
data_cutoff = 0.0001 # when to crop the upper parts of the data
side_labels = ['נ', 'ג', 'ה', 'ש']

input_data = np.array([n_nun, n_gimel, n_hei, n_shin])
posterior_params = prior_params + input_data
dist = dirichlet(posterior_params)
test_values = np.linspace(0,1,resolution)
output_pdf = np.ndarray([n_sides, resolution])
# collect distribution data
for side in range(n_sides): # test each side separately
    for index, prob in enumerate(test_values):
        # get probability for other sides not tested
        probability = (np.ones(n_sides) - prob) / (n_sides - 1)
        # replace the side tested with the probability so it all adds to one
        probability[side] = prob
        # save distribituion
        output_pdf[side, index] = dist.pdf(probability)

###
#Generate the plot
###

# convert to df for analysis and plotting
df = pd.DataFrame(output_pdf, columns = test_values, index = side_labels).T
# normalize each side's probability distribution to 1
df = df / df.sum()
# get cdf
output_cdf = df.cumsum()
# plot preprocessing
# indentify which indexes to plot by removing higher value data that is near zero
index_to_keep = df.loc[~(df < data_cutoff).all(1)].index * 100
# identify max for each side
peak_value = df.max()
peak_index = pd.Series(dtype=int) 
for c in df.columns:
    peak_index[c] = df[c][df[c] == peak_value[c]].index[0] * 100
# plot
f, ax = plt.subplots()
df.index = df.index * 100
df.plot(ax=ax)
ax.set_xlim((0,max(index_to_keep)))
ax.set_ylim((0,ax.get_ylim()[1]))
ax.set_xlabel('Chance of landing (%)')
ax.set_ylabel('Probability of that chance')
ax.set_yticks([])
# plot line showing fairness
ax.plot((100/n_sides, 100/n_sides),ax.get_ylim(), color='gray', linestyle='--')
ax.annotate(text='fair line', xy=(100/n_sides*1.025, ax.get_ylim()[1]*.90), color='gray')

# plot all four data labels near their peak
for c in df.columns:
    ax.annotate(text=c, xy=(peak_index[c], peak_value[c]*.95), ha='center',va='top')
ax.get_legend().remove()


###
#Provide qualitative assessment
###

if sum(input_data) < 25:
    print_txt = '... lacking data. Collect more data to be more certain whether or not it is biased.'
else:
    indexes = output_cdf.index
    index_of_fairline = np.argmin(np.abs(indexes-1/n_sides))
    cdf_at_fairline = output_cdf.iloc[index_of_fairline]
    diff_from_midpoints = (cdf_at_fairline - 0.5).abs()
    max_diff_from_midpoint = diff_from_midpoints.max()
    if max_diff_from_midpoint > 0.475:
        print_txt = 'almost certainty biased. Go get a better dreidel.'
    elif max_diff_from_midpoint > 0.45:
        print_txt = 'very likely biased. Probably should get a better dreidel.'
    elif max_diff_from_midpoint > 0.35:
        print_txt = 'likely biased.'
    elif max_diff_from_midpoint > 0.25:
        print_txt = 'possibly biased.'
    elif sum(input_data) < 130:
        print_txt = 'too close to call. Keep rolling.'
    elif max_diff_from_midpoint > 0.15:
        print_txt = 'a pretty accurate dreidel. Good job.'
    elif max_diff_from_midpoint > 0.10:
        print_txt = 'an accurate dreidel. Nice find.'
    elif max_diff_from_midpoint > 0.05:
        print_txt = 'a really accurate dreidel! A Hanukkah Miracle!'



body.write('### Your dreidel is ' + print_txt)

body.pyplot(f)


methods.write("""
There are multiple ways to statistically determine if a dreidel is biased. This app uses a Bayesian statical approach which accounts for our prior understanding of how bias a dreidel is likely to be and the amount of data collected. 

For the bayesian approach, one needs to describe both our prior understand and the probability function of spinning the dreidel. This app uses probability function of dice rolling is called the [Multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution) distribution. The prior understanding uses a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) distribution with a value of 5 for all the parameters, since the distribution has a peak at 25% and is broad enough to account for expected variations in dreidel fairness. The math behind this is described [here](https://stephentu.github.io/writeups/dirichlet-conjugate-prior.pdf).

The descriptions of how fair the dreidel is were arbitrarily determined and are meant for young students to more easily conceptualize the probability distribution.

Another way to determine if a dreidel is biased is called the [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test), but that was not deployed here because it doesn't account for prior understanding of how biased a dreidel is likely to be nor does it account for the amount of data collected. 
""")

misc.write("""Created by Mark Goldman, November 2024""")

