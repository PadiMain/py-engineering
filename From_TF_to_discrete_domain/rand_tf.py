import numpy as np
import matplotlib.pyplot as plt
"""
For p (Laplace variable) make a replacement 2/Td * ((z - 1)/(z + 1)) in z domain and then derive the difference equation
             1
  tf = ---------------
       0.5 s^2 + s + 1
       
Input = step from 1 s
"""


def step(x, step_time):
    in_p = [1 if n >= step_time else 0 for n in x]
    return np.array(in_p)


def rand_tf_fun(input, initial_val):
    output = initial_val
    for n in range(2, len(input)):
        output.append(0.009349989202382 * input[n-1] + 0.008746764185212 * input[n-2] + 1.800633999690388 * output[n-1] - 0.818730753077982 * output[n-2])
    return np.array(output)


def main():
    time = 10  # Stop simulation time
    fd = 10  # Sampling frequency
    x = np.arange(0, time, 1/fd)  # Simulation time
    in_put = step(x, 1)  # Input of tf
    output = rand_tf_fun(in_put, [0, 0])  # Output of tf (Initial condition=0, 0)
    plt.plot(x, output)  # Plot of output
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

