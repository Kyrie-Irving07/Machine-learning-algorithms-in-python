import tensorflow as tf
import os
import numpy as np
import scipy.io as sio
from six.moves import xrange
import scipy.interpolate as interpolate
import h5py
import time

path = 'model1.mat'
file = "C:\\Users\Dell\Desktop\Research\AAAI\program\\venv\som_result.txt"


def load_data(path, result_max=0.9, result_min=-0.9):
    data = h5py.File(path)
    logdr = data['logdr']
    ds = data['ds']
    neighbour = data['neighbour']

    L = data['L']
    L = np.transpose(L)

    v = data['v']
    v = np.transpose(v)

    vneighbour = data['VV']
    vneighbour = np.transpose(vneighbour, (1, 0))
    vneighbour = np.array(vneighbour, np.int32)

    eimat = data['ei_mat']
    eimat = np.transpose(eimat, (1, 0))

#    eivalue = data['ei_value']
#    eivalue = np.transpose(ei_value, (1, 0))

    edgenum = len(logdr[0])
    modelnum = len(logdr[0][0])
    y = load_neighbour(neighbour, edgenum)

    logdr_x = np.zeros((modelnum, edgenum, 3)).astype('float64')
    logdr_x = logdr
    logdr_x = np.transpose(logdr_x, (2, 1, 0))

    ds_x = np.zeros((modelnum, edgenum, 6)).astype('float64')
    ds_x = ds
    ds_x = np.transpose(ds_x, (2, 1, 0))

    logdrmin = logdr_x.min() - 1e-6
    logdrmax = logdr_x.max() + 1e-6

    dsmin = ds_x.min() - 1e-6
    dsmax = ds_x.max() + 1e-6

    # mapping the arguments into space [-0.9, 0.9]
    logdrnew = (result_max - result_min) * (logdr_x - logdrmin) / (logdrmax - logdrmin) + result_min
    dsnew = (result_max - result_min) * (ds_x - dsmin) / (dsmax - dsmin) + result_min
    x = np.concatenate((logdrnew, dsnew), axis=2)

    return x, y, logdrmin, logdrmax, dsmin, dsmax, modelnum, edgenum, v, vneighbour, L, eimat # , eivalue


def load_neighbour(neighbour, edges, is_padding=False):
    data = neighbour

    if is_padding == True:
        x = np.zeros((edges + 1, 4)).astype('int32')

        for i in xrange(0, edges):
            x[i + 1] = data[:, i] + 1

            for j in xrange(0, 4):
                if x[i + 1][j] == -1:
                    x[i + 1][j] = 0

    else:
        x = np.zeros((edges, 4)).astype('int32')

        for i in xrange(0, edges):
            x[i] = data[:, i]

    return x


x, y, logdrmin, logdrmax, dsmin, dsmax, modelnum, edgenum, v, vneighbour, L, eimat = load_data(path)


class somap():
    def __init__(self, eimat, vneighbour, learning_rate=0.1, weight=0.5, decay=1e5, width=20, height=20):
        self.map = np.array(np.random.uniform(-1e-5, 0.02, [width, height, np.shape(eimat)[1]]))
        self.map = np.random.normal(np.mean(eimat), np.var(eimat), size=[width, height, np.shape(eimat)[1]])
        self.range = max(width, height)
        self.vneighbour = vneighbour
        self.eimat = eimat
        self.learning_rate = learning_rate
        self.weight = weight
        self.width = width
        self.height = height
        self.decay = decay

    def get_vector(self, indice):
        iden = self.eimat[indice]
        near = iden
        for i in range(np.shape(self.vneighbour)[1]):
            if self.vneighbour[indice][i] != 0:
                newindice = self.vneighbour[indice][i] - 1
                near = np.row_stack([near, self.eimat[newindice]])
        near = np.mean(near, 0)
        iden = iden * self.weight
        near = near * (1 - self.weight)
        return iden + near

    def map_distance(self, position1, position2):
        p1 = np.array(position1)
        p2 = np.array(position2)
        assert np.shape(p1) == (2,)
        assert np.shape(p2) == (2,)
        x = np.min([np.abs(p1[0] - p2[0]), self.width - np.abs(p1[0] - p2[0])])
        y = np.min([np.abs(p1[1] - p2[1]), self.height - np.abs(p1[0] - p2[0])])
        mapdst = np.square(x) + np.square(y)
        return mapdst

    def get_rate(self, step, position, nearest):
        mapdst = self.map_distance(position, nearest)
        return self.learning_rate * np.exp(- (step / self.decay) - mapdst / np.square(self.range) * 5)

    def vector_distance(self, v1, v2):
        v = v1 - v2
        v = np.square(v)
        return np.sum(v, 0)

    def get_nearest(self, indice):
        nearest = [0, 0]
        mindst = 1000
        for i0 in range(self.width):
            for j0 in range(self.height):
                temp = self.vector_distance(self.eimat[indice], self.map[i0][j0])
                if temp < mindst:
                    mindst = temp
                    nearest = [i0, j0]
        return nearest

    def update(self, indice, step):
        nearest = self.get_nearest(indice)
        for i0 in range(self.width):
            for j0 in range(self.height):
                rate = self.get_rate(step, [i0, j0], nearest)
                # print("nearest %d, %d position %d, %d rate %f" % (nearest[0], nearest[1], i0, j0, rate))
                self.map[i0][j0] = self.map[i0][j0] + rate * (self.get_vector(indice) - self.map[i0][j0])


mysom = somap(eimat, vneighbour)

visual = []

for i in range(1000000):
    indice = np.random.randint(0, np.shape(np.array(eimat))[0] - 1)
    mysom.update(indice, i)
    if i % 500 == 0:
        print("%d steps" % i)
    if i % 2000 == 0:
        outcome = []
        visual0 = visual
        visual = np.zeros([20, 20])
        for j in range(2161):
            outcome.append([mysom.get_nearest(j)[0], mysom.get_nearest(j)[1]])
            visual[outcome[j][0]][outcome[j][1]] += 1
        if i != 0 and (visual0 == visual).all():
            print("Nothing change")
        else:
            print("Changed")
        for i0 in range(20):
            for j in range(20):
                print("%3d" % visual[i0][j], end=' ')
            print()
        with open(file, "w") as f:
            for i1 in range(np.shape(outcome)[0]):
                f.write(str(i1) + ': ')
                for i2 in range(np.shape(outcome)[1]):
                    f.write(str(outcome[i1][i2]) + '\t')
                f.write('\n')

outcome = []
visual = np.zeros([20, 20])
for j in range(2161):
    print("50000 steps final outcome:")
    outcome.append([mysom.get_nearest(j)[0], mysom.get_nearest(j)[1]])
    print("%d node : %d, %d" % (j, outcome[j][0], outcome[j][1]))
    visual[outcome[j][0]][outcome[j][1]] += 1

for i in range(20):
    for j in range(20):
        print("%3d" % visual[i][j], end=' ')
    print()
