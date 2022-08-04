from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import random as rn

"""Kalman filter on state-space model"""





def step(time, ts, x):
    for i in range(int(time/ts)):
        yield x


def main():
    """State-space representation of a plane flight with wind (standard deviation = 0.001)
    with state vector = [velocity, path angle, pitch angle, angular velocity of the pitch angle, altitude]
    input vector = [deviation from the trim value of the elevator]
    sensor standard deviation = 4
    sensor frequency = 100 hz
    simulation time = 100 s
    """
    A = np.array([[-0.06, -4.6, -5.25, 0, 0], [0.004, -0.63, 0.62, 0, 0], [0, 0, 0, 1, 0], [-0.0002, 0.5, -0.5, -0.53, 0], [-0.046, 72, 0, 0, 0]])
    B = np.array([[0], [0.02], [0], [-0.4], [0]])
    C = np.eye(5)
    Ts = 0.01
    time = 100
    q = 0.001
    r = 4
    R = np.array([[r, 0, 0, 0, 0], [0, r, 0, 0, 0], [0, 0, r, 0, 0], [0, 0, 0, r, 0], [0, 0, 0, 0, r]])
    Q = np.array([[q, 0, 0, 0, 0], [0, q, 0, 0, 0], [0, 0, q, 0, 0], [0, 0, 0, q, 0], [0, 0, 0, 0, q]])
    in_put_var = step(time, Ts, -0.1)
    # Euler integration for state-space model in discrete time
    Ad = np.eye(5) + Ts * A
    Bd = Ts * B
    Cd = C
    # Discrete state vector x
    x = np.array([[0 + rn.normalvariate(0, q)], [0 + rn.normalvariate(0, q)], [0 + rn.normalvariate(0, q)], [0 + rn.normalvariate(0, q)], [0 + rn.normalvariate(0, q)]])
    y = x + np.array([[rn.normalvariate(0, r)], [rn.normalvariate(0, r)], [rn.normalvariate(0, r)], [rn.normalvariate(0, r)], [rn.normalvariate(0, r)]])
    x_est = y
    p_est = np.array([R])
    y_t = x
    in_put_var = step(time, Ts, -0.1)

    while True:
        try:
            # Simulating flight and sensor measurements
            ud = next(in_put_var)
            x = np.matmul(Ad, x) + Bd*ud + Ts * np.matmul(Q, np.array([[rn.normalvariate(0, q)], [rn.normalvariate(0, q)], [rn.normalvariate(0, q)], [rn.normalvariate(0, q)], [rn.normalvariate(0, q)]]))
            y_temp = (np.matmul(Cd, x) + np.matmul(R, np.array([[rn.normalvariate(0, r)], [rn.normalvariate(0, r)], [rn.normalvariate(0, r)], [rn.normalvariate(0, r)], [rn.normalvariate(0, r)]])))
            y = np.append(y, y_temp, axis=1)
            y_t_temp = np.matmul(Cd, x)
            y_t = np.append(y_t, y_t_temp, axis=1)

        except StopIteration:
            break

    in_put_var = step(time, Ts, -0.1)
    for i in range(1, int(time/Ts) + 1):
        # Kalman algorithm
        # prediction
        x_pred_est = np.matmul(Ad, np.reshape(x_est[:, i - 1], (5, 1))) + Bd * next(in_put_var)
        p_pred_est = np.matmul(np.matmul(Ad, p_est[i - 1]), np.transpose(Ad)) + Q
        # update
        Kalman = np.matmul(np.matmul(p_pred_est, np.transpose(Cd)), np.linalg.inv((np.matmul(np.matmul(Cd, p_pred_est), np.transpose(C)) + R)))
        x_est = np.append(x_est, (x_pred_est + np.matmul(Kalman, (np.reshape(y[:, i], (5, 1)) - np.matmul(Cd, x_pred_est)))), axis=1)
        p_est = np.append(p_est, np.array([np.matmul((np.eye(5) - np.matmul(Kalman, Cd)), p_pred_est)]), axis=0)


    fig, ax = plt.subplots()
    xdata1, ydata1 = [], []
    ln1, = plt.plot([], [], '-')
    xdata, ydata = [], []
    ln, = plt.plot([], [], '-')
    t_s = len(list(range(0, int(time/Ts) + 1)))

    def init():
        ax.set_xlim(0, time/10)
        ax.set_ylim(min(x_est[4])-10, max(x_est[4]))
        return ln,

    def update(frame):
        xdata1.append(frame * Ts)
        ydata1.append(y[4][frame])
        ln1.set_data(xdata1, ydata1)
        xdata.append(frame*Ts)
        ydata.append(x_est[4][frame])
        ln.set_data(xdata, ydata)

        return ln1, ln

    ani = animation.FuncAnimation(fig, update, frames=t_s, init_func=init, blit=False, interval=Ts/100)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
