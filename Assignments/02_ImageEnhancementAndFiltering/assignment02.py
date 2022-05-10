import numpy as np
import matplotlib.pyplot as plt 
import imageio

# Method 1
def limiarization(I):

    T0 = int(input())
    T = T0
    T_new = T

    while T_new - T > 0.5 or T_new == T0:
        T = T_new

        # Binarizing the image using the threshold
        G1 = I[np.where(I > T)]
        G2 = I[np.where(I <= T)]
        
        # Updating T
        mu_1 = np.mean(G1)
        mu_2 = np.mean(G2)
        T_new = 0.5*(mu_1 + mu_2)

    # Creating new image 
    I_new = np.zeros(I.shape)
    I_new[np.where(I > T)] = 1

    return I_new

# Method 2
def filtering_1d(I):

    # Reading input
    n = int(input())
    c = int(n/2)
    weight = np.array(list(map(int, input().split())))

    # Preparing a copy from I as a circular array (wrap image)
    I_pad = np.copy(I.flatten())
    I_pad = np.pad(I_pad, (c,c), 'wrap')

    N = len(I.flatten())
    wn = len(weight)

    # New image
    #   multiplies the next wn elements in each iteration by the weight array
    #   I_new has the same shape as I, I_pad is larger after padding
    I_new = np.zeros(N)
    for i in range(0, N):
        I_new[i] = sum(I_pad[i:i+wn]*weight)

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
    I_pad = np.copy(I)
    I_pad = np.pad(I_pad, (c,c), 'symmetric')
    
    N = I.shape[0]

    # New image
    #   Applies the convolution 
    #   I_new has the same shape as I, I_pad is larger after padding
    #   M receives part of I_pad flattend to apply the convolution
    I_new = np.zeros( I.shape )
    wf = weight.flatten()
    for i in range(c,N+c):
        for j in range(c,N+c):
            M = I_pad[i-c:i+c+1, j-c:j+c+1].flatten()
            I_new[i-c][j-c] = (-1) * np.convolve(M, wf, mode='valid')

    # Applying limiarization
    I_new = limiarization(I_new)

    return I_new

# Method 4
def median_filter(I):
    
    n = int(input())
    c = int(n/2)

    N = I.shape[0]

    # Preparing a copy from I for the filter
    I_pad = np.copy(I)
    I_pad = np.pad(I_pad, (c,c), mode='constant', constant_values=0)
    
    # New image
    #   idx_median: index for the median value, since the matrix is flattend, the 
    #   middle index is n**2/2, n being the shape to be considered by the filter
    I_new = np.zeros(I.shape)
    idx_median = int(n**2/2)
    for i in range(c,N+c):
        for j in range(c,N+c):
            R = np.sort(I_pad[i-c:i+c+1, j-c:j+c+1].flatten())
            I_new[i-c][j-c] = R[idx_median]

    return I_new

# Normalising the new image to values between 0 and 255
def normalisation(g):
    min_value = np.min(g)
    max_value = np.max(g)

    res = (g - min_value) / (max_value - min_value) 
    return res*255


# Plot image I and I_new
def plot_images(I, I_new):

    plt.subplot(121)
    plt.imshow(I,cmap="gray")
    plt.subplot(122)
    plt.imshow(I_new,cmap="gray")
    plt.show()

# Returns the RMSE between original image I and the new image I_new
def rmse(I, I_new):
    return np.sqrt(np.sum((I - I_new)**2)/(I.shape[0] * I.shape[1]))

# Reading filename and opening image
filename = str(input()).rstrip()
I = imageio.imread(filename)

# Reading input for method to be used
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
print(rmse(I, I_new))
plot_images(I, I_new)
