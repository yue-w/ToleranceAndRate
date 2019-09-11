# =============================================================================
# import functions as f
# 
# import numpy as np
# 
# import nlopt
# 
# from pandas import DataFrame
# =============================================================================

import pandas as pd

df=pd.read_excel("Test.xlsx",sheet_name='sheet1')
print(df['noinspect_U'][0])