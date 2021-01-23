import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
 
 
np.random.seed(0)
points = np.loadtxt("E:\\SeniorYearUp\\Final\\project\\Resource\\standard.xyz")
 
fig = plt.figure(1)
plt.clf()
ax = Axes3D(fig)
#ax.scatter(points[0:3000, 0], points[0:3000, 1], points[0:3000, 2], c='r',marker='.',
           #s=10, linewidths=1, alpha=1,cmap='spectral')
#ax.scatter(points[3000:, 0], points[3000:, 1], points[3000:, 2], c='g',marker='.',
           #s=10, linewidths=1, alpha=1,cmap='spectral')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='g',marker='.',
           s=10, linewidths=1, alpha=1,cmap='spectral')           
ax.set_title("bunny")
ax.set_xlabel("1st ")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd ")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd ")
ax.w_zaxis.set_ticklabels([])
 
plt.show()