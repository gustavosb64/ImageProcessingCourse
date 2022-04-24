import numpy as np
import matplotlib.pyplot as plt 
import imageio

# Method 1
def limiarization(I):

    T0 = int(input())
    T = T0
    T_new = T

    while T_new == T0 or T_new - T > 0.5:
        T = T_new

        G1 = I[np.where(I > T)]
        G2 = I[np.where(I <= T)]
        
        mu_1 = np.mean(G1)
        mu_2 = np.mean(G2)
        T_new = 0.5*(mu_1 + mu_2)

    I_new = np.zeros(I.shape)
    I_new[np.where(I > T)] = 1

    return I_new

# Method 2
def filtering_1d(I):

    n = int(input())
    weight = np.array(list(map(int, input().split())))

    c = int(n/2)

    """
    I = np.array([ [1,2,3] , [4,5,6] , [7,8,9] ])
    weight = np.array([0.5, 0.3, 0.2])
    N = len(I)
    c = int(3/2)
    """

    # Preparing a copy from I as a circular vector (wrap image)
    I_wrap = np.copy(I.flatten())
    I_wrap = np.pad(I_wrap, (c,c), 'wrap')

    N = len(I.flatten())
    wn = len(weight)

    # New image
    I_new = np.zeros(N)

    for i in range(0, N):
        I_new[i] = sum(I_wrap[i:i+wn]*weight)

    I_new = I_new.reshape(I.shape)

    return I_new

# Method 3
def filtering_2d(I):

    # Reading input
    n = int(input())
    weight = np.zeros((n,n))
    for i in range(n):
        weight[i] = np.array(list(map(int, input().split())))

    c = int(weight.shape[0]/2)

    # Preparing a copy from I as a symmetric matrix for convolution
    I_sym = np.copy(I)
    I_sym = np.pad(I_sym, (c,c), 'symmetric')
    
    N = I.shape[0]

    # New image
    I_new = np.zeros( I.shape )

    wf = weight.flatten()
    for i in range(c,N+c):
        for j in range(c,N+c):

            M = I_sym[i-c:i+c+1, j-c:j+c+1].flatten()
            I_new[i-c][j-c] = (-1)*np.convolve(M, wf, mode='valid')

    return limiarization(I_new)

def median_filter(I):
    
    n = int(input())

    """
    n = 3
    I = np.array([ [23, 5, 39, 45, 50],
                   [70, 88, 12, 100, 110],
                   [130, 145, 159, 136, 137],
                   [19, 200, 201, 220, 203],
                   [131, 32, 133, 34, 135] ])
    """

    c = int(n/2)

    N = I.shape[0]

    I_pad = np.copy(I)
    I_pad = np.pad(I_pad, (c,c), mode='constant', constant_values=0)

    I_new = np.zeros(I.shape)
    for i in range(c,N+c):
        for j in range(c,N+c):
            R = np.sort(I_pad[i-c:i+c+1, j-c:j+c+1].flatten())
            I_new[i-c][j-c] = R[int((n**2)/2)]
#            I_new[i-c][j-c] = np.median(I_pad[i-c:i+c+1, j-c:j+c+1].flatten())

    return I_new

def normalisation(g):
    min_value = np.min(g)
    max_value = np.max(g)

    res = (g - min_value) / (max_value - min_value) 
    return res*255

def plot_and_rmse(I, I_new):
    #RMSE
    print(np.sqrt(np.sum((I - I_new)**2)/(I.shape[0] * I.shape[1])))

    plt.subplot(221)
    plt.imshow(I,cmap="gray")
    plt.subplot(222)
    plt.imshow(I_new,cmap="gray")
    plt.show()

# Reading filename and opening image
filename = str(input()).rstrip()
I = imageio.imread(filename)
method = int(input())

if method == 1:
    I_new = limiarization(I)
elif method == 2:
    I_new = filtering_1d(I)
elif method == 3:
    I_new = filtering_2d(I)
elif method == 4:
    I_new = median_filter(I)

I_new = normalisation(I_new)
plot_and_rmse(I, I_new)
