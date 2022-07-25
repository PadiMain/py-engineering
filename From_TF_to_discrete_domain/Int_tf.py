import numpy as np
import matplotlib.pyplot as plt


def input_fun(x, freq):
    in_p = [2 * np.pi * freq * n for n in x]
    return 1 * np.sin(in_p)


def Intagrate_tf_fun(input, Td, initial_val):
    output = [initial_val]
    for n in range(1, len(input)):
        output.append(Td * (input[n] + input[n-1])/2 + output[n-1])
    return np.array(output)


def main():
    freq = 10
    Time = 1/freq
    fd = 1000
    x = np.arange(0, Time, 1/fd)
    in_put = input_fun(x, freq)
    output = Intagrate_tf_fun(in_put, 1/fd, 0)
    plt.plot(x, output)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

