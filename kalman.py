import numpy as np


class KalmanFilter(object):
    def __init__(self, init_x, init_v, accel_variance):
        # mean of state gaussian random variable
        self.x = np.array([init_x, init_v])
        self.accel_variance = accel_variance

        # covariance of state gaussian random variable
        self.p = np.eye(2)

        # initialize h
        self.h = np.array([1, 0]).reshape((1, 2))

    def predict(self, delta_t):
        # use predication kalman filter equations
        # new_x = F * x
        # new_p = F * P * F.T + G * a * G.T

        # initialize f and g
        f = np.array([[1, delta_t], [0, 1]])
        g = np.array([0.5*delta_t**2, delta_t]).reshape((2, 1))

        # calculate the new_x and new_p
        new_x = np.dot(f, self.x)
        new_p = np.dot(np.dot(f, self.p), f.T) + \
            np.dot(g, g.T) * self.accel_variance**2

        self.x = new_x
        self.p = new_p

    def update(self, measured_value, measured_variance):
        # use update kalman filter equations
        # y = z - h * x
        # s = h * p * h.T + r
        # k = p * h.t * s^-1
        # x = x + k * y
        # p = (I - k * h) * p

        # initialize z and r
        z = np.array([measured_value])
        r = np.array([measured_variance])

        # calculate y, s, and k
        y = z - np.dot(self.h, self.x)
        s = np.dot(np.dot(self.h, self.p), self.h.T) + r
        k = np.dot(np.dot(self.p, self.h.T), np.linalg.inv(s))

        # calculate the new x and
        self.x = self.x + np.dot(k, y)
        self.p = np.dot((np.eye(2) - np.dot(k, self.h)), self.p)

    def covariance(self):
        return self.p

    def mean(self):
        return self.x
