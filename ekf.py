import numpy as np

class EKF:
    def __init__(self,  initial_x: np.array, 
                        # Q: np.array,
                        P_0: np.array) -> None: # returns nothing
        # mean of state Gaussian RV
        self._x = initial_x.reshape(6,1)

        # covariance of initial state (Gaussian RV)
        self._P = P_0

    def predict(self,   F: np.array, 
                        G: np.array, 
                        u: np.array, 
                        Q: np.array) -> None:

        # np.random.seed(20)

        u = u.reshape(2,1)

        # process noise
        fn = np.zeros((6,1)) # no process noise added
        
        # fn[0] = np.random.normal(0,  0.1)
        # fn[1] = np.random.normal(0,  0.1)
        # fn[2] = np.random.normal(0, 0.001)
        # fn[3] = np.random.normal(0,  0.01)
        # fn[4] = np.random.normal(0,  0.01)
        # fn[5] = np.random.normal(0, 0.001)
        
        # prediction equations
        new_x = F.dot(self._x) + G.dot(u) + fn
        new_P = F.dot(self._P).dot(F.T) + Q

        self._P = new_P
        self._x = new_x


    def update(self,    H: np.array, 
                        meas_value: np.array, 
                        meas_variance: np.array): 

        # np.random.seed(20)

        # noisy measurement
        y = meas_value.reshape(6,1)

        # measurement variance
        R = meas_variance.reshape(6,6)

        # noise of measurement prediction
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
        y_hat = H.dot(self._x) + hn

        # innovation
        innov = y - y_hat
        
        
        C_xy = self._P.dot(H.T)
        C_yy = H.dot(self._P).dot(H.T) + R
        
        # kalman gain
        K = C_xy.dot(np.linalg.inv(C_yy))

        # state and covariance update
        new_x = self._x + K.dot(innov)
        new_P = self._P - K.dot(C_xy.T)

        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P


    @property
    def state(self) -> np.array:
        return self._x