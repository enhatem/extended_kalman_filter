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
def evolution(X, U, m, Ixx, dt):
    g = 9.81 # m/s^2

    y       = X[0] + X[3]*dt
    z       = X[1] + X[4]*dt
    phi     = X[2] + X[5]*dt
    vy      = X[3] + (    - U[0] / m ) * np.sin(X[2]) * dt
    vz      = X[4] + ( -g + U[0] / m ) * np.cos(X[2]) * dt
    phi_dot = X[5] + U[1] / Ixx * dt

    return np.array([y, z, phi, vy, vz, phi_dot]) 