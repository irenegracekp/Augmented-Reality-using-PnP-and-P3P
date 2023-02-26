import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    
    pixal_h = np.concatenate((pixels[:,:], np.ones((pixels.shape[0],1))), axis = 1)
    R_wc_inv = np.linalg.inv(R_wc)
    t_wc_new = -np.matmul(R_wc_inv, t_wc).reshape(3,1)

    transform = np.concatenate([R_wc_inv[:,0:-1], t_wc_new], axis = 1)
    H = np.matmul(K, transform)
    H_inv = np.linalg.inv(H)
    
    Pw = np.zeros((pixels.shape[0], 3))

    for i in range(pixels.shape[0]):
        Pw[i, :] = np.transpose(np.matmul(H_inv, pixal_h[i, :].T))
        Pw[i, :] = Pw[i, :]/Pw[i, -1]
        Pw[i, -1] = 0 



    ##### STUDENT CODE END #####
    return Pw
