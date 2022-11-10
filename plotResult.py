import matplotlib.pyplot as plt
import numpy as np

dataArray = np.genfromtxt('result.csv', delimiter=',', names=True)

plt.figure()
for col_name in dataArray.dtype.names:
    if col_name != 'Iteration':
        plt.plot(dataArray[col_name], label=col_name)
plt.legend()
plt.show()