import pandas as pd
import numpy as np

result = pd.read_csv('result_three.csv')
length = result.shape[0]
lo  = 0

while(lo < length):
    t_str = result['id'][lo]
    t_lo = t_str.find('.')
    result['id'][lo] = t_str[0:t_lo]
    lo = lo + 1
result.to_csv('result_doggood.csv')