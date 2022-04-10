import numpy as np
import imageio as img
import random 
import matplotlib.pyplot as plt

# Defined functions
f1 = lambda x,y: x*y + 2*y
f2 = lambda x,y,Q: abs(np.cos(x/Q) + 2*np.sin(y/Q) )
f3 = lambda x,y,Q: abs(3*(x/Q) - np.power(y/Q, 1/3))
f4 = lambda S: random.random()

def f5(f, C, S):
    
    x, y = 0, 0
    for i in range(1 + C**2):
        f[x][y] = 1
        
        dx, dy = random.randint(-1,1), random.randint(-1,1)
        x = (x + dx)%C
        y = (y + dy)%C

    return f

def normalisation(g, B):

    min_value = np.min(g)
    max_value = np.max(g)
    res = (g - min_value)/(max_value - min_value)
    return res * (2**B - 1)

def RSE(g, R, C, N):

    res = 0.0
    print("R",len(R))
    print("g",len(g))

    size = len(R)
    for i in range(size):
        for j in range(size):
            res += (float(g[i][j]) - float(R[i][j]))**2

    return np.sqrt(res)

def function_choice(f, x, y, C, Q, N, B, S, function):

    if f == 1:
        return f1(x,y)
    if f == 2:
        return f2(x,y,Q)
    if f == 3:
        return f3(x,y,Q)
    if f == 4:
        return f4(S)
    
def downsampling(g, C, N):
    
    print(C, N, C/N)

    step = int( C/N ) 
    print("step", step)
    
    #Dimensões da matriz final
    if (C/step > int(C/step)):
        f_size = int(C/step)+1
    else:
        f_size = int(C/step)

    f = np.zeros( (f_size, f_size ) )
    fi, fj = 0, 0
    for i in range(0, C, step):
        for j in range(0, C, step):
            f[fi][fj] = g[i][j]
            fj += 1
        fi += 1
        fj = 0
    
    return f 

# Reading input
filename = str(input()).rstrip()
C = int(input())
f_choice = int(input())
Q = int(input())
N = int(input())
B = int(input())
S = int(input())

#Inicializando a seed a ser usada em random
random.seed(S)

f = np.zeros( (C, C) )

#A função 5 retorna a matriz completa, ao passo que as outras calculam cada pixel
if f_choice == 5:
    f = f5(f, C, S)
else:
    for i in range (C):
        for j in range (C):
            f[i][j] = function_choice(f_choice, i, j, C, Q, N, B, S, f)

g = downsampling(f, C, N)

g = normalisation(g, B)
#g = g.astype(np.uint8) >> (8 - B)

# Opening file
R = np.load(filename).astype(np.uint8)
print(RSE(g,R, C, N))

# Plotando imagens
plt.figure()
fig = plt.subplot(2, 2,1)
plt.imshow(g, cmap="gray")
fig = plt.subplot(2, 2,2)
plt.imshow(R, cmap="gray")
fig = plt.subplot(2, 2,3)
plt.imshow(f, cmap="gray")

plt.show()
