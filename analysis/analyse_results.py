#%% Prelims
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:57:35 2018

@author: AlexR
"""

import matplotlib.pyplot as plt
import json
import os
import matplotlib as mpl
import pandas as pd

import numpy as np

def in_json(name):
    with open(name, "r") as fh:
        return np.transpose(np.array(json.load(fh))[:, 1:])  #skip timestamp

#%%

hier_hi_loss = in_json("run_35-tag-hi_loss_1.json")
hier_hi_score = in_json("run_35-tag-hi_score_1.json")
hier_lo_loss = in_json("run_35-tag-lo_loss_1.json")
hier_lo_score = in_json("run_35-tag-lo_score_1.json")

biped_score = in_json("run_ddpg_biped_fast-tag-score_1.json")
#%%
ant_score = in_json("run_tensorboard_ant-tag-score_1.json")

#%% Plots

plt.figure()
plt.plot(
    hier_hi_loss[0],
    np.where(hier_hi_loss[1] == 0, np.nan, hier_hi_loss[1]),
    label="High Level Loss",
    color='tab:pink')
plt.plot(
    hier_lo_loss[0],
    np.where(hier_lo_loss[1] == 0, np.nan, hier_lo_loss[1]),
    label="Low Level Loss",
    color='tab:cyan')
plt.legend()
plt.savefig('hier_loss.png')

plt.figure()
plt.plot(*hier_hi_score, label="High Level Score", color='tab:pink')
plt.plot(*hier_lo_score, label="Low Level Score", color='tab:cyan')
plt.legend()
plt.savefig('hier_score.png')

plt.figure()
plt.plot(*biped_score, label="Bipedal Walker Score", color='tab:cyan')
plt.legend()
plt.savefig('biped_score.png')

#%%
plt.figure()
plt.plot(ant_score[1, :2000], label="Ant Score", color='tab:cyan')
plt.legend()
plt.savefig('ant_score.png')

#%%
ant_cut = ant_score[:, ant_score[0] <= 2000]
ant_mask = np.unique(ant_cut[0, :2000], return_index=True)[1]
ant_mask = ant_mask[ant_mask < 2000]
plt.figure()
plt.plot(*ant_cut[:, ant_mask], label="Ant Score", color='tab:cyan')
plt.legend()
plt.savefig('ant_score.png')

#%%
