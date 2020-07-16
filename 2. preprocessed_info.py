#preprocessed_info
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./csv/preprocessed.csv")

print(data.info())
print(data.describe())

data.hist(bins=50, figsize=(20,15))
plt.show()

data.plot(kind="scatter", x="Lng", y="Lat", alpha=0.1, label="totalPrice", c="totalPrice", cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()

import numpy as np
# Make the plot
plt.hexbin(x=data['Lng'], y=data['Lat'],C=data['totalPrice'],reduce_C_function=np.std,gridsize=50, cmap=plt.get_cmap("jet"))
plt.show()
 
