import numpy as np


class KalmanFilter2D(object):
    def __init__(self,
                 init_x=0,
                 init_y=0,
                 init_x_vel=0,
                 init_y_vel=0,
                 x_accel_var=0,
                 y_accel_var=0):
        # mean of state gaussian random variable
        self.x = np.array([init_x, init_y, init_x_vel, init_y_vel])
        self.x_accel_var = x_accel_var
        self.y_accel_var = y_accel_var

        # covariance of state gaussian random variable
        self.p = np.eye(4) * 0.2

        # initialize h
        self.h = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [
                          0, 0, 0, 0], [0, 0, 0, 0]]).reshape((4, 4))

    def predict(self, delta_t):
        # use predication kalman filter equations
        # new_x = F * x
        # new_p = F * P * F.T + Q

        # initialize f and g
        f = np.eye(4)
        f[0][2] = f[1][3] = delta_t

        q = np.eye(4)
        q[0][0] = q[1][1] = 0.001
        q[2][2] = q[3][3] = 0

        # calculate the new_x and new_p
        new_x = np.dot(f, self.x)
        new_p = np.dot(np.dot(f, self.p), f.T) + q

        self.x = new_x
        self.p = new_p
        return self.x

    def update(self, measured_x_pos, measured_y_pos, measured_x_vel, measured_y_vel):
        # use update kalman filter equations
        # y = z - h * x
        # s = h * p * h.T + r
        # k = p * h.t * s^-1
        # x = x + k * y
        # p = (I - k * h) * p

        # initialize z and r
        z = np.array([measured_x_pos, measured_y_pos,
                      measured_x_vel, measured_y_vel])
        r = np.eye(4)
        r[0][0] = r[1][1] = r[2][2] = r[3][3] = 0.1

        # calculate y, s, and k
        y = z - np.dot(self.h, self.x)
        s = np.dot(np.dot(self.h, self.p), self.h.T) + r
        k = np.dot(np.dot(self.p, self.h.T), np.linalg.inv(s))

        # calculate the new x and
        self.x = self.x + np.dot(k, y)
        self.p = np.dot((np.eye(4) - np.dot(k, self.h)), self.p)
        return self.x
