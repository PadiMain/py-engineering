import numpy
import matplotlib.pyplot as plt
import math


class Measurer:
    def __init__(self, tf_num, tf_den, sig, fd=0.1, time=1, amp_delta=0.2, w_delta=50):
        self.fd = fd
        self.sig = sig
        self.tf_num = tf_num
        self.tf_den = tf_den
        self.time = time
        self.amp_delta = amp_delta
        self.w_delta = w_delta

    def signal(self):
        for t in range(int(self.time / self.fd)+1):
            sig = self.sig(t * self.fd) + self.amp_delta * math.sin(2 * math.pi * self.w_delta * t * self.fd)
            if 300 <= t <= 400:
                sig = 0
            yield sig

    def comp_output(self, input, start):
        x_1 = input[0]
        x_2 = input[1]
        y_1 = start[0]
        y_2 = start[1]
        a1 = self.tf_num[0]
        a2 = self.tf_num[1]
        b1 = self.tf_den[0]
        b2 = self.tf_den[1]
        b3 = self.tf_den[2]
        return (a1/b1) * x_1 + (a2/b1) * x_2 - (b2/b1) * y_1 - (b3/b1) * y_2


class Acc(Measurer):
    pass


class Gps(Measurer):
    def __init__(self, tf_num, tf_den, sig, fd=0.1, time=1, amp_delta=0.2, w_delta=5):
        super().__init__(tf_num, tf_den, sig, fd, time, amp_delta, w_delta)


def main():
    fd = 0.01
    time = 10
    acc = Acc([4.998833329196413e-05, 4.997667070806499e-05], [1, -1.999299745117823, 0.999300244942843], lambda t: 0.5*t+10, fd, time)
    gps = Gps([7.000049405102042e-04, -6.995051154902039e-04], [1, -1.999299745117823, 0.999300244942843], lambda t: (0.5*t*t*t)/6+(10/2)*t*t, fd, time)
    acelerometr = acc.signal()
    GPS = gps.signal()
    y0_acc = 0
    y1_acc = 0
    x1_acc = 0
    y0_gps = 0
    y1_gps = 0
    x1_gps = 0

    comp = [y0_acc + y0_gps]
    while True:

        try:
            x0_acc = next(acelerometr)

            x0_gps = next(GPS)

        except:
            break
        y_acc = acc.comp_output([x0_acc, x1_acc], [y0_acc, y1_acc])
        x1_acc = x0_acc
        y1_acc = y0_acc
        y0_acc = y_acc
        y_gps = gps.comp_output([x0_gps, x1_gps], [y0_gps, y1_gps])
        x1_gps = x0_gps
        y1_gps = y0_gps
        y0_gps = y_gps

        comp.append(y0_acc + y0_gps)


    plt.plot(numpy.arange(0, time+2*fd, fd), comp)
    plt.plot(numpy.arange(0, time + 2*fd, fd), [0.5*t*t*t/6+10/2*t*t for t in numpy.arange(0, time + 2*fd, fd)])
    # plt.plot(numpy.arange(0, time + 2*fd, fd), numpy.array([0.5*t*t*t/6+10/2*t*t for t in numpy.arange(0, time + 2*fd, fd)]) - numpy.array(comp))
    plt.grid()
    plt.show()



if __name__ == '__main__':
    main()