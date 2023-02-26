import numpy as np
import math

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    ##### STUDENT CODE END #####
    X = Pc
    
    u0 = K[0, 2]
    v0 = K[1, 2]    
    f = K[0, 0]
    
    j1 = np.linalg.inv(K)@np.array([X[1,0], X[1,1], 1]).T
    j1 = j1/np.linalg.norm(j1)
    j2 = np.linalg.inv(K)@np.array([X[2,0], X[2,1], 1]).T
    j2 = j2/np.linalg.norm(j2)
    j3 = np.linalg.inv(K)@np.array([X[3,0], X[3,1], 1]).T
    j3 = j3/np.linalg.norm(j3)

    a = np.linalg.norm(Pw[2,:] - Pw[3,:])
    b = np.linalg.norm(Pw[3,:] - Pw[1,:])
    c = np.linalg.norm(Pw[1,:] - Pw[2,:])

    Calpha = np.dot(j2,j3)
    Cbeta = np.dot(j3,j1)
    Cgamma = np.dot(j1,j2)

    A0 = (1 + (a**2 - c**2)/b**2)**2 - (4* a**2 * Cgamma**2)/b**2
    A1 = 4*( ((c**2 - a**2)/b**2)*(1 + (a**2 - c**2)/b**2)*Cbeta + ((2* a**2 * Cgamma**2 * Cbeta)/b**2) - ((1 -  ((a**2 + c**2)/b**2))* Calpha*Cgamma) )
    A2 = 2*( ((a**2 - c**2)/b**2)**2 - 1 + 2*(((a**2 - c**2)/b**2)**2)*Cbeta**2 + 2*((b**2 - c**2)/b**2)*Calpha**2 - 4*((a**2 + c**2)/b**2)*Calpha*Cbeta*Cgamma + 2*((b**2 - a**2)/b**2)*Cgamma**2)
    A3 = 4*( ((a**2 - c**2)/b**2)* ( 1 - ((a**2 - c**2)/b**2) )*Cbeta - ( 1 - ((a**2 + c**2)/b**2))*Calpha*Cgamma + 2*c**2*Calpha**2*Cbeta/b**2 )
    A4 = (((a**2 - c**2)/b**2 - 1)**2 - 4*c**2*Calpha**2/b**2 )

    p = [A4, A3, A2, A1, A0]
    sols = np.roots(p)
    solutions = sols[np.isreal(sols)].real

    u = []
    s1 = []
    s2 = []
    s3 = []
    R_list = []
    t_list = []
    B = np.zeros((len(solutions),3))
    norms = np.zeros((len(solutions),1))

    for i in range(len(solutions)):
        v = solutions[i]
        u.append(( (-1 + ((a**2 - c**2)/b**2))*v**2 - 2*((a**2 - c**2)/b**2)*v*Cbeta + 1 + ((a**2 - c**2)/b**2) )/ (2*(Cgamma - v*Calpha)))
        d1 = math.sqrt(c**2/(1+ u[i]**2 - 2*u[i]*Cgamma))
        s1.append(math.sqrt(c**2/(1+ u[i]**2 - 2*u[i]*Cgamma)))
        d2 =  u[i]*s1[i]
        s2.append(u[i]*s1[i])
        d3 = solutions[i]*s1[i]
        s3.append(solutions[i]*s1[i])

        p1_1 = j1*s1[i]
        p1_2 = j2*s2[i]
        p1_3 = j3*s3[i]
        p_cam = np.array([p1_1.T, p1_2.T, p1_3.T])
        
        R, t = Procrustes(Pw[1:,:], p_cam)
        R_list.append(R)
        t_list.append(t)
        
        B[i] = np.matmul(K, np.matmul(R, Pw[0,:].T) + t)
        B[i] = B[i]/B[i,2]
        norms[i] = np.linalg.norm(B[i, 0:2] - Pc[0,:])

    id = np.argmin(norms)
    R = R_list[id]
    t = t_list[id]

    R = np.linalg.inv(R)
    t = -np.matmul(R,t)
    
    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####

    Y_mean = np.mean(Y, axis=0).T
    X_mean = np.mean(X, axis=0).T

    Y_val = (Y - Y_mean ).T
    X_val = (X - X_mean).T
    H = np.matmul(Y_val, X_val.T)

    [U, S , Vt ] = np.linalg.svd(H) 
    d = np.linalg.det(np.matmul(U, Vt)) 
    matrix = np.eye(U.shape[1])
    matrix[-1][-1] = d
    R = np.matmul(U, np.matmul(matrix, Vt))
    t = Y_mean.T - np.matmul(R, X_mean.T)

    ##### STUDENT CODE END #####
    return R, t

