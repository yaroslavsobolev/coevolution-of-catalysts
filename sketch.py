gaussian_center = 0.85

if __name__ == '__main__':
    # use numpy to plot a gaussian curve in matplotlib
    import numpy as np
    import matplotlib.pyplot as plt

    # define the gaussian function
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    # # define the x-axis
    # x_axis = np.linspace(-10, 10, 120)
    #
    # # plot the gaussian function
    # plt.plot(x_axis, gaussian(x_axis, gaussian_center, 1), color='black')
    # plt.xlabel('Conditions space')
    # plt.ylabel('Yield')
    # plt.xticks([])
    # plt.axhline(y=0.05)
    # plt.ylim(-0.05, 1.1)
    # plt.annotate('Detection threshold', xy=(-10, 0.07), color='C0')
    # plt.fill_between(x=x_axis, y1=0.05, y2=gaussian(x_axis, gaussian_center, 1), where=gaussian(x_axis, gaussian_center, 1) > 0.05, color='C0', alpha=0.2)
    # # plt.show()
    # plt.savefig('gauss1.png')
    #
    # x2 = np.linspace(-10, 10, 5)
    # plt.scatter(x2, gaussian(x2, gaussian_center, 1), color='C1')
    # plt.title('Initial blind sampling')
    # plt.savefig('gauss2.png')
    # # plt.show()
    #
    # plt.title('Gradient descent')
    # x2 = np.linspace(-0.1, gaussian_center, 5)
    # plt.scatter(x2, gaussian(x2, gaussian_center, 1), color='C2')
    # plt.savefig('gauss3.png')
    # plt.show()





    # N = 100
    # X, Y = np.mgrid[0.01:1:complex(0, N), 0.01:1:complex(0, N)]
    # Z = gaussian(Y, 0.2+X/2, 0.1*(1-X+0.1))
    # Z[X<0.85] = np.nan
    #
    # fig, ax0 = plt.subplots(1)
    # plt.ylabel('Free coordinates of condition space')
    # plt.xlabel('Coordinates for which we prefer certain value \n (i.e. substrate structure, product structure)')
    # plt.tight_layout()
    # c = ax0.pcolor(X, Y, Z, cmap='plasma')
    # cbar = fig.colorbar(c, ax=ax0, label='Yield')
    # plt.axvline(x=0.93, color='white', linestyle='--')
    # plt.savefig('map1.png')
    #
    # plt.show()



    N = 100
    X, Y = np.mgrid[0.01:1:complex(0, N), 0.01:1:complex(0, N)]

    Z = gaussian(Y, 0.2+X/2, 0.1*(1-X+0.1))
    # Z[X<0.85] = np.nan

    fig, ax0 = plt.subplots(1)
    plt.ylabel('Free coordinates of condition space')
    plt.xlabel('Coordinates for which we prefer certain value \n (i.e. substrate structure, product structure)')
    plt.tight_layout()
    c = ax0.pcolor(X, Y, Z, cmap='plasma')
    cbar = fig.colorbar(c, ax=ax0, label='Yield')
    plt.axvline(x=0.93, color='white', linestyle='--')
    plt.axvline(x=0.1, color='C2', linestyle='--')
    plt.axvline(x=0.35, color='C2', linestyle='--')
    plt.axvline(x=0.6, color='C2', linestyle='--')
    plt.axvline(x=0.80, color='C2', linestyle='--')
    plt.savefig('map2.png')

    ys = np.linspace(0.1, 0.9, 5)
    xs = 0.1 * np.ones_like(ys)
    plt.scatter(xs, ys, color='C2')
    plt.savefig('map3.png')

    plt.show()

