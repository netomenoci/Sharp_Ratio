#!/usr/bin/env python
# coding: utf-8

# ## In this notebook we'll generate data samples to train our algorithm

# In[1]:


from Functions_with_moment import *


# In[2]:


#features_random = compute_parameters_random(800, number_of_series=60, number_of_permutations=60)


# In[6]:


features_mix = compute_parameters_mix(3, v_i = 2.1, v_f = 11, size_v = 10, number_of_series = 10 , permutation_factor = 0.5 , series_size_i = 50, series_size_f = 300, sr_i = 1e-2, sr_f =1)
##features_random = compute_parameters_for(v_i = 2.1, v_f = 10, size_v = 3, number_of_series = 60 ,  permutation_factor = 0.5 , series_size_i = 50, series_size_f = 300, series_size_step = 25, sr_i = 1e-2, sr_f =1,size_sr = 2)


# In[7]:


df = pd.DataFrame.from_records(features_mix, columns = ['c' , 'v', 'n', 'R-', 'R+', 'c_moment', 'std_moment'])


# In[8]:


df


# In[14]:


features_random = compute_parameters_random(100000, v_i = 2.1, v_f = 30 , number_of_series=60, number_of_permutations=60)


# In[3]:


#df = pd.DataFrame.from_records(features_random, columns = ['c' , 'v', 'n', 'R-', 'R+'])
df = pd.DataFrame.from_records(features_random, columns = ['c' , 'v', 'n', 'R-', 'R+', 'c_moment'])


# In[5]:


df.to_csv('features_random_100k_60_60_v_2_30_with_c_moment', index = False)


# In[13]:


df = pd.read_csv('features_random_100k_60_60')


# In[4]:


df.head()


# In[15]:


df.corr()


# In[ ]:


features_for = compute_parameters_for(v_i = 11, v_f = 30, size_v = 40, number_of_series = 60 , number_of_permutations = 60 , series_size_i = 50, series_size_f = 100, series_size_step = 4,sr_i = 1e-1, sr_f =2,size_sr = 20)

