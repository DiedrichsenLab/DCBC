#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Feb 25 10:15:48 2021

Plotting DCBC curves

INPUTS:     struct: DCBC evaluation result
OUTPUT:     The figure of within- and between parcels correlation curve

Author: Da Zhi
'''
import numpy as np
import matplotlib.pyplot as plt
from DCBC.eval_DCBC import scan_subdirs


def plot_single(within, between, subjects, maxDist=35, binWidth=1,
                within_color='k', between_color='r'):
    fig = plt.figure()

    # Obtain basic info from evaluation result
    numBins = int(np.floor(maxDist / binWidth))
    num_sub = len(subjects)
    x = np.arange(0, maxDist, binWidth) + binWidth / 2

    y_within = within.reshape(num_sub, -1)
    y_between = between.reshape(num_sub, -1)

    plt.errorbar(x, y_within.mean(0), yerr=y_within.std(0), ecolor=within_color, color=within_color,
                 elinewidth=0.5, capsize=2, linestyle='dotted', label='within')
    plt.errorbar(x, y_between.mean(0), yerr=y_between.std(0), ecolor=between_color, color=between_color,
                 elinewidth=0.5, capsize=2, linestyle='dotted', label='between')

    plt.legend(loc='upper right')
    plt.show()


def plot_wb_curve(T, path, sub_list=None, hems='all', within_color='k', between_color='r'):
    fig = plt.figure()

    # Obtain basic info from evaluation result T
    bin_width = [value for key, value in T.items()][0]['binWidth']
    max_dist = [value for key, value in T.items()][0]['maxDist']
    k = len([value for key, value in T.items()][0]['corr_within'])
    x = np.arange(0,max_dist,bin_width) + bin_width/2

    # if hems is 'all' and any([x for x in T.keys() if 'L' in x]) and any([x for x in T.keys() if 'R' in x]):
    #     # subjectsDir = [x for x in T.keys()]
    #     pass
    # elif hems is 'L' or 'R' and any([x for x in T.keys() if hems in x]):
    #     # subjectsDir = [x for x in T.keys() if hems in x]
    #     pass
    # else:
    #     raise TypeError("Input hemisphere's data has not been found!")

    if sub_list is not None:
        subjects_dir = sub_list
    else:
        subjects_dir = scan_subdirs(path)

    y_within, y_between = np.empty([1, k]), np.empty([1, k])
    for sub in subjects_dir:
        data = [value for key, value in T.items() if sub in key]
        if len(data) == 2 and hems is 'all':
            within = (np.asarray(data[0]["corr_within"]) + np.asarray(data[1]["corr_within"])) / 2
            between = (np.asarray(data[0]["corr_between"]) + np.asarray(data[1]["corr_between"])) / 2
        elif len(data) == 1 and data[0]["hemisphere"] is hems:
            within = data[0]["corr_within"]
            between = data[0]["corr_between"]
        else:
            raise Exception("Incomplete DCBC evaluation. Missing result of %s." % sub)

        y_within = np.vstack((y_within, within))
        y_between = np.vstack((y_between, between))
        sub_list = T.keys()

    y_within = np.delete(y_within, 0, axis=0)
    y_between = np.delete(y_between, 0, axis=0)

    plt.errorbar(x, y_within.mean(0), yerr=y_within.std(0), ecolor=within_color, color=within_color, label='within')
    plt.errorbar(x, y_between.mean(0), yerr=y_between.std(0), ecolor=between_color, color=between_color, label='between')

    plt.legend(loc='upper right')
    plt.show()



def plot_DCBC(T, color='r'):
    # todo: to plot DCBC value
    print('working on this function')


# T = dict()
# T['s02_L']={"binWidth": 1,
#             "maxDist": 35,
#             "hemisphere": 'L',
#             "num_within": [1,2,3,4,5,6,7,8,9,10],
#             "num_between": [0,1,2,3,4,5,6,7,8,9],
#             "corr_within": np.random.rand(35),
#             "corr_between": np.random.rand(35),
#             "weight": [0,1,2,3,4,5,6,7,8,9],
#             "DCBC": 10}
# T['s02_R']={"binWidth": 1,
#             "maxDist": 35,
#             "hemisphere": 'R',
#             "num_within": [1,2,3,4,5,6,7,8,9,10],
#             "num_between": [0,1,2,3,4,5,6,7,8,9],
#             "corr_within": np.random.rand(35),
#             "corr_between": np.random.rand(35),
#             "weight": [0,1,2,3,4,5,6,7,8,9],
#             "DCBC": 10}
# T['s03_L']={"binWidth": 1,
#             "maxDist": 35,
#             "hemisphere": 'L',
#             "num_within": [1,2,3,4,5,6,7,8,9,10],
#             "num_between": [0,1,2,3,4,5,6,7,8,9],
#             "corr_within": np.random.rand(35),
#             "corr_between": np.random.rand(35),
#             "weight": [0,1,2,3,4,5,6,7,8,9],
#             "DCBC": 10}
# T['s03_R']={"binWidth": 1,
#             "maxDist": 35,
#             "hemisphere": 'R',
#             "num_within": [1,2,3,4,5,6,7,8,9,10],
#             "num_between": [0,1,2,3,4,5,6,7,8,9],
#             "corr_within": np.random.rand(35),
#             "corr_between": np.random.rand(35),
#             "weight": [0,1,2,3,4,5,6,7,8,9],
#             "DCBC": 10}
#
#
# if __name__ == '__main__':
#     plot_wb_curve(T, path='data', sub_list=['s02', 's03'])


