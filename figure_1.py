import numpy as np
import matplotlib.pyplot as plt

from kalman import KalmanFilter

NUM_STEPS = 100
MEASUREMENT_EVER_STEPS = 5

initial_x = 0.0
initial_v = 1.0
low_a_variance = 0.2
high_a_variance = 2

delta_t = 0.1

kf_low_a = KalmanFilter(initial_x, initial_v, low_a_variance)
kf_high_a = KalmanFilter(initial_x, initial_v, high_a_variance)

means_low = []
covs_low = []
means_high = []
covs_high = []

real_xs = []
real_vs = []

real_x = 0.0
real_v = 0.9
meas_variance = 0.1**2

for i in range(NUM_STEPS):
    covs_low.append(kf_low_a.covariance())
    covs_high.append(kf_high_a.covariance())
    means_low.append(kf_low_a.mean())
    means_high.append(kf_high_a.mean())

    real_x = real_x + delta_t * real_v

    kf_low_a.predict(delta_t)
    kf_high_a.predict(delta_t)
    if i != 0 and i % MEASUREMENT_EVER_STEPS == 0:
        curr_x = real_x + np.random.randn() * np.sqrt(meas_variance)
        kf_low_a.update(curr_x, meas_variance)
        kf_high_a.update(curr_x, meas_variance)

    real_xs.append(real_x)
    real_vs.append(real_v)

plt.figure(1)

plt.subplot(2, 1, 1)
plt.title("Position")
plt.plot([m[0] for m in means_low], 'r')  # position
plt.plot(real_xs, 'b')

plt.plot([m[0] - 2*np.sqrt(cov[0, 0])
          for m, cov in zip(means_low, covs_low)], 'r--')  # standard deviation
plt.plot([m[0] + 2*np.sqrt(cov[0, 0])
          for m, cov in zip(means_low, covs_low)], 'r--')  # standard deviation

plt.subplot(2, 1, 2)
plt.title("Velocity")
plt.plot([m[1] for m in means_low], 'r')
plt.plot(real_vs, 'b')

plt.plot([m[1] - 2*np.sqrt(cov[1, 1])
          for m, cov in zip(means_low, covs_low)], 'r--')  # standard deviation
plt.plot([m[1] + 2*np.sqrt(cov[1, 1])
          for m, cov in zip(means_low, covs_low)], 'r--')  # standard deviation

plt.figure(2)
plt.subplot(2, 1, 1)
plt.title("Position")
plt.plot([m[0] for m in means_high], 'r')  # position
plt.plot(real_xs, 'b')

plt.plot([m[0] - 2*np.sqrt(cov[0, 0])
          for m, cov in zip(means_high, covs_high)], 'r--')  # standard deviation
plt.plot([m[0] + 2*np.sqrt(cov[0, 0])
          for m, cov in zip(means_high, covs_high)], 'r--')  # standard deviation

plt.subplot(2, 1, 2)
plt.title("Velocity")
plt.plot([m[1] for m in means_high], 'r')
plt.plot(real_vs, 'b')

plt.plot([m[1] - 2*np.sqrt(cov[1, 1])
          for m, cov in zip(means_high, covs_high)], 'r--')  # standard deviation
plt.plot([m[1] + 2*np.sqrt(cov[1, 1])
          for m, cov in zip(means_high, covs_high)], 'r--')  # standard deviation

plt.show()
