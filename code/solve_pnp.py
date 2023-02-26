from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    

    H = est_homography(Pw[:,0:2], Pc)
    H = H/H[-1,-1]
    print(H)
    K_inv_H = np.matmul(np.linalg.inv(K), H)
    H = K_inv_H
    h1 = H[:,0]
    h2 = H[:,1]
    h3 = np.cross(h1, h2)

    h = np.vstack((h1, h2, h3))
    h = h.T

    [U, S , Vt ] = np.linalg.svd(h) 
    d = np.linalg.det(np.matmul(U, Vt)) 
    matrix = np.eye(U.shape[1])
    matrix[-1][-1] = d
    R = np.matmul(U, np.matmul(matrix, Vt))
    R = R.T
    t = -np.matmul(R, (K_inv_H[:,2]/(np.linalg.norm(K_inv_H[:,0]))))
    ##### STUDENT CODE END #####

    return R, t

