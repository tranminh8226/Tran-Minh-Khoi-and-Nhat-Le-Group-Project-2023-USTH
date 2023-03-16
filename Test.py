from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as clear_output
#import six.moves as urllib
#import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

df = pd.read_csv(r"I:\VNL data\Data\Alldata.csv")

 
# applying the method
count_nan = df.isnull()
print(count_nan)



result = df['GHI'].isnull().to_numpy().nonzero()
print(result)
# printing the number of values present
# in the column

plt.figure(figsize=(16,8))
plt.plot(df.Wind1)
plt.show()