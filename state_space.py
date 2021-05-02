import numpy as np

# quadrotor parameters
m = 0.027 / 2 # m=27g
Ixx = 1.657171e-05

def getA(phi,u1):
    A = np.array([ 
        [0, 0,                      0,                      1, 0, 0],
        [0, 0,                      0,                      0, 1, 0],
        [0, 0,                      0,                      0, 0, 1],
        [0, 0,    -(u1/m)*np.cos(phi),                      0, 0, 0],
        [0, 0,    -(u1/m)*np.sin(phi),                      0, 0, 0],
        [0, 0,                      0,                      0, 0, 0]
    ])
    return A

def getB(phi):
    B = np.array([ 
        [0,              0],
        [0,              0],
        [0,              0],
        [-np.sin(phi)/m, 0],
        [ np.cos(phi)/m, 0],
        [0,          1/Ixx]
    ])
    return B