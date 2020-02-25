import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from matplotlib import pyplot as plt
from sympy.utilities.iterables import multiset_permutations
from itertools import permutations
import random

def compute_R(p):
    '''
    Computes the number of minima and maximuma (records) of a given random walk p
    '''

    R_plus = len(set(np.maximum.accumulate(p)))
    R_minus = len(set(np.minimum.accumulate(p)))

    return (R_minus, R_plus)

def compute_R_mean(number_of_series , number_of_permutations , series_size, v, c):
    '''
    For a given student distribution (with v degrees of freedom and expected value c we
    return the average number of records )
    '''
    std = np.sqrt(v/(v-2))
    SR = c/std
    R_plus_vector = []
    R_minus_vector = []
    c_moment = []
    std_moment = []
    for series in range(number_of_series):
        r = (student_t.rvs(v, size=series_size))/std + c
        c_moment.append(r.mean(dtype = np.float64))
        std_moment.append(r.std(ddof=1, dtype = np.float64)) #ddof = 1 divides by n-1 instead of n (get the unibased estimator)
        R_plus_sum = 0
        R_minus_sum = 0

        for permutation in range(number_of_permutations):
            sample_permutation = np.random.permutation(r)
            sample_permutation = np.cumsum(sample_permutation) #Create a random walk from the steps

            R_plus_sum += compute_R(sample_permutation)[1]
            R_minus_sum += compute_R(sample_permutation)[0]

        R_plus_mean = R_plus_sum/number_of_permutations
        R_minus_mean = R_minus_sum/number_of_permutations

        R_plus_vector.append(R_plus_mean)
        R_minus_vector.append(R_minus_mean)


    R_plus_mean = np.mean(R_plus_vector)
    R_minus_mean = np.mean(R_minus_vector)
    c_moment_value = np.mean(c_moment, dtype = np.float64)
    std_moment_value = np.mean(std_moment, dtype = np.float64)

    return (R_minus_mean , R_plus_mean, c_moment_value, std_moment_value)


def compute_tuple_R_SR(v, number_of_series = 50 , number_of_permutations = 50 , series_size = 400, sr_i = 1e-1, sr_f =2,size_sr = 100):
    sharp_ratio_list = np.linspace(sr_i, sr_f, size_sr)
    #std = np.sqrt(v/(v-2))
    R_plus_list = []
    R_minus_list = []
    for sharp_ratio in sharp_ratio_list:
        c = sharp_ratio ##*std ## from the definition used for sharp ratio
        R_minus_mean , R_plus_mean = compute_R_mean(number_of_series , number_of_permutations , series_size, v, c)

        R_plus_list.append(R_plus_mean)
        R_minus_list.append(R_minus_mean)
    return (sharp_ratio_list, R_minus_list, R_plus_list)

def plot_records(sharp_ratio_list,R_minus_list,R_plus_list):
    fig = plt.figure()

    ax = fig.add_axes([0,0,1,1])

    ax.plot(sharp_ratio_list,R_plus_list, label="R+")
    ax.plot(sharp_ratio_list, R_minus_list, label="R-")
    ax.plot(sharp_ratio_list,(np.array(R_plus_list) - np.array(R_minus_list)), label="R0")

    ax.grid(True)
    ax.legend()

def compute_tuple_R_series_size(v, c, number_of_series = 50 , number_of_permutations = 50 , series_size_i = 30,series_size_f = 400, step = 3):
    std = np.sqrt(v/(v-2))
    R_plus_list = []
    R_minus_list = []
    series_size_list = range(series_size_i,series_size_f, step) #np.linspace(10,200,1)
    for series_size in series_size_list:
        R_minus_mean , R_plus_mean = compute_R_mean(number_of_series , number_of_permutations , series_size, v, c)
        R_plus_list.append(R_plus_mean)
        R_minus_list.append(R_minus_mean)

    return (series_size_list,R_minus_list,R_plus_list )

def compute_parameters_for(v_i = 2.1, v_f = 10, size_v = 200, number_of_series = 60 ,  permutation_factor = 0.5 , series_size_i = 50, series_size_f = 300, series_size_step = 4, sr_i = 1e-2, sr_f =1,size_sr = 20):
    sharp_ratio_list = np.linspace(sr_i, sr_f, size_sr)
    v_list = np.linspace(v_i, v_f, size_v)
    series_size_list = range(series_size_i,series_size_f, series_size_step)
    features = []
    for sharp_ratio in sharp_ratio_list:
        for v in v_list:
            for series_size in series_size_list:

                c = sharp_ratio ##*std ## from the definition used for sharp ratio
                number_of_permutations = int(permutation_factor*series_size) + 1
                R_minus_mean , R_plus_mean, c_mean, std_mean = compute_R_mean(number_of_series , number_of_permutations , series_size, v,c)
                features.append([c, v, series_size,R_minus_mean,R_plus_mean, c_mean, std_mean])
    return features

def compute_parameters_random(number_of_points, v_i = 2.1, v_f = 11, number_of_series = 10 , permutation_factor = 0.5 , series_size_i = 50, series_size_f = 300, sr_i = 1e-2, sr_f =1):
    '''
    This function is about 7 times slower than the compute_parameters_for to generate the same number of points
    '''
    
    features = []
    for i in range(number_of_points):
        c = random.uniform(sr_i,sr_f)
        v =  random.uniform(v_i,v_f)
        series_size = random.randint(series_size_i,series_size_f + 1)
        number_of_permutations = int(permutation_factor*series_size) + 1
        R_minus_mean , R_plus_mean, c_mean, std_mean = compute_R_mean(number_of_series , number_of_permutations , series_size, v, c)
        features.append([c, v, series_size,R_minus_mean,R_plus_mean, c_mean, std_mean])
    return features

def compute_parameters_mix(number_of_points, v_i = 2.1, v_f = 11, size_v = 200, number_of_series = 10 , permutation_factor = 0.5 , series_size_i = 50, series_size_f = 300, sr_i = 1e-2, sr_f =1):
    '''
    Mix between the random generation and the for generation. We range v between v_i and v_f usize size_v point
    '''
    v_list = np.linspace(v_i, v_f, size_v)
    features = []
    for v in v_list:
        for i in range(number_of_points):
            c = random.uniform(sr_i,sr_f)
            series_size = random.randint(series_size_i,series_size_f + 1)
            number_of_permutations = int(permutation_factor*series_size) + 1
            R_minus_mean , R_plus_mean, c_mean, std_mean = compute_R_mean(number_of_series , number_of_permutations , series_size, v, c)
            features.append([c, v, series_size,R_minus_mean,R_plus_mean, c_mean, std_mean])
    return features

