#TRIAD SGD
import math
from math import cos, sin, atan, asin
import numpy as np


def angle2dcm(psi, teta, phi):
    r11 = cos(teta) * cos(psi)
    r12 = cos(teta) * sin(psi)
    r13 = -sin(teta)
    r21 = sin(phi) * sin(teta) * cos(psi) - cos(phi) * sin(psi)
    r22 = sin(phi) * sin(teta) * sin(psi) + cos(phi) * cos(psi)
    r23 = sin(phi) * cos(teta)
    r31 = cos(phi) * sin(teta) * cos(psi) + sin(phi) * sin(psi)
    r32 = cos(phi) * sin(teta) * sin(psi) - sin(phi) * cos(psi)
    r33 = cos(phi) * cos(teta)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def dcm2angel(R):
    psi = atan(R[0][1]/R[0][0]) * 180 / math.pi
    teta = -asin(R[0][2]) * 180 / math.pi
    phi = atan(R[1][2]/R[2][2]) * 180 / math.pi
    return psi, teta, phi


def norm(vect):
    return (vect[0] ** 2 + vect[1] ** 2 + vect[2] ** 2) ** (1/2)


def df(R, M, Mo):
    Q = np.matmul(Mo, np.transpose(M))
    lm = (1/2) * (np.eye(3, 3) + np.matmul(Q, np.transpose(Q)))
    dr = np.transpose(Q) - np.matmul(R, lm)
    return dr


PSI = 10 * math.pi / 180
TETA = 20 * math.pi / 180
PHI = 30 * math.pi / 180

R = angle2dcm(PSI, TETA, PHI)
eo = np.array([2, 0, 0])
eon = eo/norm(eo)
so = np.array([0, 4, 0])
son = so/norm(so)
mo = np.cross(eon, son)
no = np.cross(eon, mo)
Mo = np.transpose(np.array([eon, mo, no]))
#----------------
en = np.matmul(R, eon)
sn = np.matmul(R, son)
m = np.cross(en, sn)
n = np.cross(en, m)
M = np.transpose(np.array([en, m, n]))
#----------------
EPOCHS = 1000
R_sgd = np.zeros((3, 3))
nt = 0.01
for i in range(EPOCHS):
    dR = df(R_sgd, M, Mo)
    R_sgd = R_sgd + nt * dR

print(dcm2angel(R_sgd))