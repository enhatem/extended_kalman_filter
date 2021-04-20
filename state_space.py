import numpy as np

# quadrotor parameters
m = 0.027 / 2 # m=27g
Ixx = 1.657171e-05

def getA(yaw,u1):
    A = np.array([ 
        [0, 0,                      0,                      1, 0, 0],
        [0, 0,                      0,                      0, 1, 0],
        [0, 0,                      0,                      0, 0, 1],
        [0, 0,    -(u1/m)*np.cos(yaw),                      0, 0, 0],
        [0, 0,    -(u1/m)*np.sin(yaw),                      0, 0, 0],
        [0, 0,                      0,                      0, 0, 0]
    ])
    return A

def getB(yaw):
    B = np.array([ 
        [0,              0],
        [0,              0],
        [0,              0],
        [-np.sin(yaw)/m, 0],
        [ np.cos(yaw)/m, 0],
        [0,          1/Ixx]
    ])
    return B