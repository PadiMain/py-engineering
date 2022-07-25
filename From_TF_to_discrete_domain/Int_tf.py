import numpy as np
import matplotlib.pyplot as plt
"""
For p (Laplace variable) make a replacement 2/Td * ((z - 1)/(z + 1)) in z domain and then derive the difference equation
"""


def input_fun(x, freq):
    in_p = [2 * np.pi * freq * n for n in x]
    return 1 * np.sin(in_p)


def integrate_tf_fun(input, Td, initial_val):
    output = [initial_val]
    for n in range(1, len(input)):
        output.append(Td * (input[n] + input[n-1])/2 + output[n-1])
    return np.array(output)


def main():
    freq = 10  # Sine frequency
    time = 1/freq  # Stop simulation time (1 period of signal)
    fd = 1000  # Sampling frequency
    x = np.arange(0, time, 1/fd)  # Simulation time
    in_put = input_fun(x, freq)  # Input of integrator
    output = integrate_tf_fun(in_put, 1/fd, 0)  # Output of integrator (integration with Initial condition=0)
    plt.plot(x, output)  # Plot of output
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

