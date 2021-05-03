import numpy as np
from state_space import evolution

class EKF:
    def __init__(self,  initial_x: np.array, 
                        # Q: np.array,
                        P_0: np.array,
                        m: float, 
                        Ixx: float,
                        dt: float) -> None: # returns nothing
        # mean of state Gaussian RV
        self._x = initial_x.reshape(6,1)

        # covariance of initial state (Gaussian RV)
        self._P = P_0

        # mass of the drone
        self._m = m
        
        # inertia of the drone
        self._Ixx = Ixx

        # sample time of the extended kalman filter
        self._dt = dt

    def predict(self,   A: np.array, 
                        B: np.array, 
                        u: np.array, 
                        Q_alpha: np.array,
                        Q_beta: np.array) -> None:

        # np.random.seed(20)

        u = u.reshape(2,1)

        # process noise
        # fn = np.zeros((6,1)) # no process noise added
        
        # fn[0] = np.random.normal(0,  0.1)
        # fn[1] = np.random.normal(0,  0.1)
        # fn[2] = np.random.normal(0, 0.001)
        # fn[3] = np.random.normal(0,  0.01)
        # fn[4] = np.random.normal(0,  0.01)
        # fn[5] = np.random.normal(0, 0.001)
        
        # prediction equations
        # new_x = F.dot(self._x) + G.dot(u) + fn
        new_x = evolution(self._x, u, self._m, self._Ixx, self._dt)
        # new_P = F.dot(self._P).dot(F.T) + Q
        new_P = A @ self._P @ A.T + B @ Q_beta @ B.T + Q_alpha
        # new_P = A.dot(self._P).dot(A.T) + B.dot(Q_beta).dot(B.T) + Q_alpha

        self._P = new_P
        self._x = new_x


    def update(self,    C: np.array, 
                        meas: np.array, 
                        meas_variance: np.array): 

        # np.random.seed(20)

        # noisy measurement
        y = meas.reshape(6,1)

        # measurement variance
        Q_gamma = meas_variance.reshape(6,6)

        # noise of the measurement prediction (no noise added)
        hn = np.zeros((6,1))
        '''
        hn[0] = np.random.normal(0,0.01)
        hn[1] = np.random.normal(0,0.01)
        hn[2] = np.random.normal(0,0.01)
        hn[3] = np.random.normal(0,0.01)
        hn[4] = np.random.normal(0,0.01)
        hn[5] = np.random.normal(0,0.01)
        '''
        # measurement prediction
        y_hat = C @ self._x + hn

        # innovation
        innov = y - y_hat
        
        K = self._P @ C.T @ np.linalg.inv( C @ self._P @C.T + Q_gamma)
        new_x = self._x + K @ innov
        new_P = ( np.eye(len(new_x)) - K @ C) @ self._P
        # C_xy = self._P @ C.T
        # C_yy = C @ self._P @ C.T + Q_gamma
        
        # kalman gain
        #K = C_xy @ np.linalg.inv(C_yy)

        # state and covariance update
        # new_x = self._x + K @ innov
        # new_P = self._P - K @ C_xy.T

        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P


    @property
    def state(self) -> np.array:
        return self._x