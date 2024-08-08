import numpy as np
import matplotlib.pyplot as plt



def plot_torus(precision, c, a):
    U = np.linspace(0, 2*np.pi, precision)
    V = np.linspace(0, 2*np.pi, precision)
    U, V = np.meshgrid(U, V)
    X = (c+a*np.cos(V))*np.cos(U)
    Y = (c+a*np.cos(V))*np.sin(U)
    Z = a*np.sin(V)
    return X, Y, Z


x, y, z = plot_torus(100, 2, 1)

fig = plt.figure()



ax = plt.axes(projection='3d')
ax.set_box_aspect(aspect = (3,3,1))
xdata=[]
ydata=[]
zdata=[]
points4 = [np.pi/1 * i for i in range(4)]
points16 = [np.pi/2 * i for i in range(4)]
points64 = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
points256 = [0, np.pi/8, np.pi/4, np.pi/4+np.pi/8, np.pi/2, np.pi/2+np.pi/8, 3*np.pi/4, 3*np.pi/4+np.pi/8, np.pi, np.pi+np.pi/8, 5*np.pi/4, 5*np.pi/4+np.pi/8, 3*np.pi/2, 3*np.pi/2+np.pi/8, 7*np.pi/4, 7*np.pi/4+np.pi/8]
points1024 = [np.pi/16 * i for i in range(32)]
points1096 = [np.pi/18 * i for i in range(36)]
points = points64
for a in points:
    for b in points:
        xdata.append((2+np.cos(b))*np.cos(a))
        ydata.append((2+np.cos(b))*np.sin(a))
        zdata.append(np.sin(b))
print(len(xdata))
ax.scatter3D(xdata,ydata,zdata, cmap='Greens')
ax.plot_surface(x, y, z, antialiased=True, color='orange', alpha=0.5)
plt.show()