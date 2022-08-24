# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 22:18:37 2022

@author: christian.shi
"""

import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt

arr_Data = []

path = "C:/Users/christian.shi/Desktop/dispense verif/test"
# C:/Users/christian.shi/Desktop/dispense verif/test
tempdir = os.getcwd()

# def read_txt(file_path):
#     with open(file_path, 'r') as f:
#         print(f.read())
for file in os.listdir():
    if file.endswith(".txt"):
        
        file_path = f"{path}\{file}"
        
    with open(file_path, 'r') as f:
        # print(f.read())
        lines = f.readlines()
        for line in lines:
            index = line.find("Measurement(weight_g")
            if(index > 1):
                result = line[index+21:index+26]
               
                result = result.replace(" ","")
                result = result.replace(",", "")
                result = result.replace(":", "")
                arr_Data.append(result) 
                
print(arr_Data)
tempdir = tempdir + "\\output.csv"
f = open(tempdir,'w')
for data in arr_Data:
    if(float(data) > 0.1):
        text = data + '\n'
        f.write(text)
f.close()

WTdata = r"C:\Users\christian.shi\Desktop\dispense verif\test\output.csv"

dataF = pd.read_csv(WTdata, header=None, names=[''], dtype=None)
#print(dataF)
dispense_mean = dataF.mean()
print(dispense_mean)
bins = np.arange(0.95, 1.05, 0.005)
# print(bins)
# fig1 = plt.figure()
# plt.hist(dataF, bins, label='dispense weight')
# plt.show()


N = dataF.count()
dof = N-1
# print(dof)
p = 0.99
confi = 0.95
chi_critical = stats.chi2.ppf(1-confi, df=dof )
sd = dataF.std()
z_critical = abs(stats.norm.ppf((1+p)/2))
interval = sd * z_critical * math.sqrt(dof * (1+(1/N))/chi_critical)
lower = np.array(dispense_mean - interval, dtype=float)
upper = np.array(dispense_mean + interval, dtype=float)
tolerance_interval = (lower, upper)
print("tolerance interval:")
print(tolerance_interval)


fig1, ax = plt.subplots()
ax.hist(dataF, bins)
# plt.plot(bins, stats.norm.pdf(bins, dispense_mean, sd))
plt.axvline(0.95, linestyle= '--', linewidth=1, color='g', label='DIR, +/-5%')
plt.axvline(1.05, linestyle= '--', linewidth=1, color='g')
plt.axvline(lower, linestyle= '-', linewidth=3, color='r', label='tolerance interval')
plt.axvline(upper, linestyle= '-', linewidth=3, color='r')
plt.legend(loc="upper right")
# plotTitle = 'ADP Stress Test Dispenses (n = ' + str(len(arr_TrimmedData)) + ')'
# ax.set_title(plotTitle)
ax.set_xlabel('dispense weight (g)')
ax.set_ylabel('Dispense Count')

# pd.DataFrame(dataF).hist(bins=10, range=(0.95, 1.05), figsize=(9,9))
# print( stats.skew(dataF))

# =============================================================================
# # plt.axvline(avg*0.95, linestyle= '--', linewidth=1,color='g')
# # plt.axvline(avg*1.05, linestyle= '--', linewidth=1,color='g')
# # plt.axvline(avg*0.85, linestyle= '-', linewidth=1,color='r')
# # plt.axvline(avg*1.15, linestyle= '-', linewidth=1,color='r')
# # plt.axvline(avg, linestyle= '-', linewidth=1,color='k')
# # fig5.savefig(os.path.join(savepath,'Figure5.png'),bbox_inches='tight')
# =============================================================================
