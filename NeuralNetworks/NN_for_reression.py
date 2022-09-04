import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, GRU, Input, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow
import csv


def main():

    inp = 10
    with open('time.csv', "r") as f:
        reader = csv.reader(f)
        time = list(reader)
    time = np.array([float(time[0][k]) for k in range(len(time[0][:])) if k % 2 == 0])

    with open('val.csv', "r") as f:
        reader = csv.reader(f)
        x = list(reader)
    x = np.array([float(x[0][k]) for k in range(len(x[0][:])) if k % 2 == 0])

    X = np.array([np.diag([x[i + k] for k in range(inp)]) for i in range(len(x) - inp)])
    Y = np.array([x[i] for i in range(inp, len(x))])

    print(X.shape, Y.shape, sep='\n')

    model = Sequential()
    model.add(Input((inp, inp)))
    model.add(GRU(64, return_sequences=True, recurrent_regularizer="l2"))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(GRU(32, recurrent_regularizer="l2"))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=Adam(0.001), metrics="accuracy")

    # model.fit(X, Y, batch_size=8, epochs=15)
    # model.save_weights('model.hdf5')

    model.load_weights('model.hdf5')


    yy = list()
    l = x[7000 - inp: 7000]

    start = np.diag(l)
    for i in range(7000, 10000):
        pred = model.predict(np.expand_dims(start, axis=0))
        yy.append(pred)

        l = np.append(l, pred[0], axis=0)

        start = np.diag(l[-1 - inp:-1])

    # print([yy[i][0][0] for i in range(len(yy))])

    plt.plot(time, x)
    plt.plot(time[7000:10000], [yy[i][0][0] for i in range(len(yy))])
    plt.show()


if __name__ == "__main__":
    main()