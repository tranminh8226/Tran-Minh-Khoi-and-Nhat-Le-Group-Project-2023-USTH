from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as clear_output
#import six.moves as urllib
#import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

df = pd.read_csv (r'D:\USTH\DLR\data\data_all.csv')  


print(df.WinDir.hist(bins=20))

plt.figure(figsize=(9,8))
plt.plot(df.Wind)
plt.show()