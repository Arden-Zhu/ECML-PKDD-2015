# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:54:37 2019

@author: arden.zhu
"""

import pickle
import csv
import calendar
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator, FormatStrFormatter
from scipy.interpolate import spline
from IPython.core.display import display_html
from keras.models import load_model
from code2.utils import np_haversine, density_map, get_clusters, plot_embeddings
from code2.data import load_data
from code2.training import start_new_session, process_features, create_model

# Display plots inline
# %matplotlib inline

# Fix random seed for reproducibility
np.random.seed(42)
data=load_data()

############################
from code2.training import full_train
full_train(n_epochs=10, batch_size=200, save_prefix='kickoff')

'''
Mean shift
https://en.wikipedia.org/wiki/Mean_shift
Clustering
Consider a set of points in two-dimensional space. Assume a circular window 
centered at C and having radius r as the kernel. Mean shift is a hill climbing
 algorithm which involves shifting this kernel iteratively to a higher density
 region until convergence. Every shift is defined by a mean shift vector. The 
 mean shift vector always points toward the direction of the maximum increase 
 in the density. At every iteration the kernel is shifted to the centroid or 
 the mean of the points within it. The method of calculating this mean depends 
 on the choice of the kernel. In this case if a Gaussian kernel is chosen instead 
 of a flat kernel, then every point will first be assigned a weight which will 
 decay exponentially as the distance from the kernel's center increases. 
 At convergence, there will be no direction at which a shift can accommodate
 more points inside the kernel.
'''