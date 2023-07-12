import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def simpleaxis(ax=None):
    '''removes the top and right boundaries from pyplot plots for aesthetic reasons'''
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def f(x, y):
    """
    Ackley function.
    $f(x, y)=-20 \exp \left[-0.2 \sqrt{0.5\left(x^2+y^2\right)}\right] -\exp [0.5(\cos 2 \pi x+\cos 2 \pi x)]+e+20$
    Reference:
    Ackley, D. H. (1987) "A connectionist machine for genetic hillclimbing", Kluwer Academic Publishers, Boston MA.
    :param x: float
    :param y: float
    :return: float
    """
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
        0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.exp(1) + 20

# # plot this function for x and y between -5 and +5 and show as color map
# x = y = np.linspace(-10, 10, 1000)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='plasma')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

def f2(x):
    return f(x, 0)

fig, axarr = plt.subplots(2, sharex=True, figsize=(6, 8))

xs = np.linspace(-10, 10, 1000)
ys = f2(xs)
T = 0.1
prob = np.exp(-ys/T)
prob = prob/np.max(prob)
ax = axarr[0]
ax.plot(xs, -ys)
simpleaxis(ax)
ax.set_ylabel('Cost function, a.u.')

# annotate the maximum at 0, 0 with arrow make arrow narrow
ax.annotate('', xy=(0, 0), xytext=(3, 0),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            )

ax.annotate('Global maximum', xy=(0, 0), xytext=(3, -0.34))

# annotate local maximum at (-0.96, -2.6) with arrow
ax.annotate('Local maxima', xy=(-0.96, -2.6), xytext=(-8.3, -3.3), textcoords='data'
            )
xytext = (-4, -3)
ax.annotate('', xy=(-0.94, -2.6), xytext=xytext, textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            )

ax.annotate('', xy=(-1.94, -4.79), xytext=xytext, textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            )

ax.annotate('', xy=(-2.97, -6.89), xytext=xytext, textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            )

ax.text(-0.15, 1, 'A', transform=ax.transAxes,
         fontsize=16, fontweight='bold', va='top', ha='left')
# plt.plot(xs, prob)
# plt.show()

# plot this function for x and y between -5 and +5 and show as color map
def P(x, T):
    return np.exp(-(f2(x)/T))

x = np.linspace(-10, 10, 1000)
# y = np.concatenate((np.logspace(-3, -1, 300), np.logspace(-1, 1.2, 100)))
y = np.logspace(-1, 1.7, 400)
X, Y = np.meshgrid(x, y)
Z = P(X, Y)

# use prolormesh to show 2d function

# ax = fig.add_subplot(111)
ax = axarr[1]
ax.pcolormesh(X, Y, Z, cmap='viridis')
ax.set_xlabel('Coordinate $x$')
ax.set_ylabel('Temperature $T$')
ax.set_yscale('log')
ax.text(-0.15, 1, 'B', transform=ax.transAxes,
         fontsize=16, fontweight='bold', va='top', ha='left')

plt.tight_layout()
fig.savefig('figures/annealing_1.png', dpi=300)
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('T')
# ax.set_zlabel('z')
# plt.show()