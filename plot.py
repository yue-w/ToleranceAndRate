# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 17:10:19 2019

@author: wyue
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel("U.xlsx",sheet_name='sheet1')

nonins_U = df['noinspect_U']
ins_U = df['inspect_U']
beta_prt1 = df['part1']
beta_prt2 = df['part2']
beta_prt3 = df['part3']

#print(nonins_U)
#print(ins_U)

ratio = np.linspace(0.5,4,16)


fig, ax1 = plt.subplots()
ax1.scatter(ratio, nonins_U, label="Scenario 1", color='r' )
ax1.scatter(ratio, ins_U, label="Scenario 2", color='b' ) #marker="x",
ax1.set_ylabel('U') 

ax1.set_xlabel('Ratio')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel(r'limit of $\gamma$')  # we already handled the x-label with ax1
ax2.scatter(ratio, beta_prt1)
ax2.scatter(ratio, beta_prt2)
ax2.scatter(ratio, beta_prt3)
ax2.tick_params(axis='y')



ax1.legend()
ax1.grid(True)

plt.show()

fig.savefig(fname='fig',dpi=300)
fig.savefig('3dPlot.tif')