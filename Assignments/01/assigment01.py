import numpy as np
import random 
#import matplotlib.pyplot as plt

# Defined functions
f1 = lambda x,y: x*y + 2*y
f2 = lambda x,y,Q: abs(np.cos(x/Q) + 2*np.sin(y/Q) )
f3 = lambda x,y,Q: abs(3*(x/Q) - np.power(y/Q, 1/3))
f4 = lambda : random.random()

# Function 5: randomwalk
def f5(f, C, S):

    x, y = 0, 0
    for i in range(1 + C**2):
        f[x][y] = 1
        
        dx, dy = random.randint(-1,1), random.randint(-1,1)
        x = (x + dx)%C
        y = (y + dy)%C

    return f

def normalisation(g):
    min_value = np.min(g)
    max_value = np.max(g)

    res = (g - min_value) / (max_value - min_value) 
    return res*255

# Root Squared Error
def RSE(g, R):

    return np.sqrt(np.sum((g - R)**2))

def function_choice(f, x, y, Q):

    if f == 1:
        return f1(x,y)
    if f == 2:
        return f2(x,y,Q)
    if f == 3:
        return f3(x,y,Q)
    if f == 4:
        return f4() 
    
def downsampling(f, C, N):
    
    step = int(C/N) 
    g = np.zeros( (N, N) )
    for i in range(N):
        for j in range(N):
            g[i][j] = f[i*step][j*step]

    return g

# Reading input
filename = str(input()).rstrip()
C = int(input())
f_choice = int(input())
Q = int(input())
N = int(input())
B = int(input())
S = int(input())

# Initializing the seed for random functions
random.seed(S)

f = np.zeros( (C, C) )

# Function 5 returns the full array, while the other ones return the value in each pixel
if f_choice == 5:
    f = f5(f, C, S)
else:
    for i in range (C):
        for j in range (C):
            f[i][j] = function_choice(f_choice, i, j, Q)

g = downsampling(f, C, N)
g = normalisation(g)

# Quantization via bitwise shift
g = g.astype(np.uint8) >> (8 - B)

# Opening file
R = np.load(filename).astype(np.uint8)

print(RSE(g,R))

# Ploting images
plt.figure()
fig = plt.subplot(2, 2,1)
plt.imshow(g, cmap="gray")
fig = plt.subplot(2, 2,2)
plt.imshow(R, cmap="gray")
fig = plt.subplot(2, 2,3)
plt.imshow(f, cmap="gray")

plt.show()
