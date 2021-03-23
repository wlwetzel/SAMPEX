import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt
x = np.linspace(-150 , 150 , 200) * m.pi / 180
y = np.linspace(-150 , 150 , 200) * m.pi / 180

xx , yy = np.meshgrid(x,y)

vals =np.arctan( np.sqrt(np.tan(xx)**2 + np.tan(yy)**2))

# plt.contourf(vals)
# plt.show()
time = pd.Timedelta(20,"milli")
print(time)
print(2*time)
