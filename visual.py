
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fn = 'pre_att/pw'

da = np.genfromtxt(fn)[1:]

filter = list(range(10)) + [14, 19, 24]
da1 = da[:, filter]

da2 = pd.DataFrame(da1, columns=[str(i+1) for i in filter])

da2.boxplot()

x = np.linspace(0, 100, 150)
plt.plot(x, ((5 + x) ** 0.8) / ((5 + 1) ** 0.8))

plt.plot(x,x**0.9)
plt.plot(x,x)

