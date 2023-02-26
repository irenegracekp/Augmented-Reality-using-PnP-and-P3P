import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out
    what X and Y should be.
    Input:
        X: 4x2 matrix of (x,y) coordinates
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####
    ax = np.zeros((4,9))
    ay = np.zeros((4,9))
    A = np.zeros((8,9))
    k = 0
    for i in range(4):
        ax[i] = [-X[i][0], -X[i][1], -1, 0, 0, 0, X[i][0]*X_prime[i][0], X[i][1]*X_prime[i][0], X_prime[i][0]]
        ay[i] = [0, 0, 0, -X[i][0], -X[i][1], -1, X[i][0]*X_prime[i][1], X[i][1]*X_prime[i][1], X_prime[i][1]]
        A[k] = ax[i]
        k += 1
        A[k] = ay[i]
        k += 1

    [U, S , Vt ] = np.linalg.svd(A) 

    h = Vt[-1]
    H = np.reshape(h, (3,3))

    ##### STUDENT CODE END #####

    return H
