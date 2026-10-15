import numpy as np
import random as r
import matplotlib.pyplot as plt
import sys
import math
from tqdm import tqdm

# simlulation parameters
SIZE_X = 128
SIZE_Y = 128
SUBGRIDSIZE = 4
MAX_P = float(SUBGRIDSIZE*SUBGRIDSIZE*4)
DENSITY = 0.50

lattice = np.zeros((SIZE_Y, SIZE_X), dtype=np.byte)
buffer_lattice = np.zeros((SIZE_Y, SIZE_X), dtype=np.byte)

def init_lattice():
    n = 0
    for y in range(SIZE_Y):
        for x in range(SIZE_X):
            for d in range(3):  # Looping over direction states.
                if r.random() < DENSITY:
                    n += 1
                    lattice[y][x] += pow( 2, d )
    add_square()
    print("Created "+str(n)+" particles out of "+str(4*SIZE_X*SIZE_Y))
    print("Expecting around "+str(4*SIZE_X*SIZE_Y*DENSITY))

def add_square():
    SSIZE = 0.15
    l = int(0.5*(SIZE_X - SSIZE*SIZE_X))
    for x in range( l, l+int(SSIZE*SIZE_X) ):
        for y in range( l, l+int(SSIZE*SIZE_X) ):
            lattice[y][x] = 15
            buffer_lattice[y][x] = 15

def nthBit( n, i):
    mask = [1, 2, 4, 8]
    return ( n & mask[i] ) != 0

def setBufferBit(y, x, n, v):
        y %= SIZE_Y
        x %= SIZE_X
        num = buffer_lattice[y][x]
        buffer_lattice[y][x] ^= (-v ^ num) & (1 << n)

def propagate():
    for y in range(SIZE_Y):
        for x in range(SIZE_X):
            n = lattice[y][x]
            if nthBit(n, 0):   #Up
                if y == 0:
                    setBufferBit(y, x, 0, 0)
                else:
                    setBufferBit(y-1, x, 0, 1 )
            if nthBit(n, 1):   #Right
                if x == SIZE_X-1:
                    setBufferBit(y, x, 1, 0)
                else:
                    setBufferBit(y, x+1, 1, 1 )
            if nthBit(n, 2):   #Down
                if y == SIZE_Y-1:
                    setBufferBit(y, x, 2, 0)
                else:
                    setBufferBit(y+1, x, 2, 1 )
            if nthBit(n, 3):   #Left
                if x == 0:
                    setBufferBit(y, x, 3, 0)
                else:
                    setBufferBit(y, x-1, 3, 1 )
            lattice[y][x] = 0

def resolveCollisions():
    for y in range(SIZE_Y):
        for x in range(SIZE_X):
            n = lattice[y][x]
            if n == 10:
                lattice[y][x] = 5
            elif n == 5:
                lattice[y][x] = 10

def update():
    global lattice, buffer_lattice
    propagate()
    buffer_lattice, lattice = lattice, buffer_lattice
    resolveCollisions()

init_lattice()
for i in tqdm(range(1,500)):
    plt.imsave('temp/'+str(i)+'.jpg',lattice)
    update()