# Student: Gustavo 
# USP number: 
# Course code: SCC0251
# Year/semestre: 2022/1st semester
# Title: Assignment04 - Image Restoration

import numpy as np
import matplotlib.pyplot as plt
import imageio

# Given function to calculate PSF
def get_motion_psf(
        dim_x: int, dim_y: int, degree_angle: float, num_pixel_dist: int = 20)-> np.ndarray:

    psf = np.zeros((dim_x, dim_y))
    center = np.array([dim_x-1, dim_y-1])//2
    radians = degree_angle/180*np.pi
    phase = np.array([np.cos(radians), np.sin(radians)])
    for i in range(num_pixel_dist):
        offset_x = int(center[0] - np.round_(i*phase[0]))
        offset_y = int(center[1] - np.round_(i*phase[1]))
        psf[offset_x, offset_y] = 1 
    psf /= psf.sum()
 
    return psf 

# Gaussian filter
def gaussian_filter (k, sigma):
    arx = np.arange((-k // 2 ) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx , arx)
    filt = np.exp(-(1/2)*(np.square(x) + np.square(y)) / np.square(sigma))
    return filt/np.sum(filt)

# Normalize the image between new_min and new_max
def normalization(g, new_min, new_max):
    min_value = np.min(g)
    max_value = np.max(g)

    res = ( (g - min_value) * ( ( new_max - new_min ) / (max_value - min_value) ) ) + new_min

    return res

# Convolving in frequency domain
def convolve(arr1, arr2):

    arr1_fft = np.fft.fft2(arr1)
    arr2_fft = np.fft.fft2(arr2)

    res = np.multiply(arr1_fft, arr2_fft)
    res = np.real(np.fft.ifft2(res))

    return res

# Returns the RMSE between original image I and the new image I_new
def rmse(I, I_new):
    return np.sqrt(np.sum((I - I_new)**2)/(I.shape[0] * I.shape[1]))

# Constrained Least Squares Method
def clsq(g, h, gamma):

    # Laplacian operator
    laplacian = np.array([ [0,-1,0],
                           [-1,4,-1],
                           [0,-1,0] ])
    # Padding
    width_ax0 = int( (g.shape[0]-laplacian.shape[0]) / 2 )
    width_ax1 = int( (g.shape[1]-laplacian.shape[1]+1) / 2 ) 
    lap_padded = np.pad(laplacian, [width_ax0, width_ax1]) 

    # Passing matrices to frequency domain
    H = np.fft.fft2(h)
    G = np.fft.fft2(g)
    P = np.fft.fft2(lap_padded) 

    # Applying filter 
    H_conj = np.conjugate(H)
    F = ( H_conj / (np.abs(H)**2 + gamma*np.abs(P)**2) ) * G

    # Reversing fft
    F = np.real(np.fft.ifft2(F))
    return np.clip(F, 0, 255).astype(np.uint8)

# Richardson-Lucy Method
def rl(I, psf, n_steps):

    eps = 1e-7
    O_k = np.full(I.shape, 1)
    for i in range(n_steps):

        conv1 = convolve(O_k, psf) 
        frac = np.divide(I, conv1+eps) 

        conv2 = convolve(frac, np.flip(psf))
        O_k = np.multiply(O_k, conv2)

    O_k = np.fft.fftshift(O_k)
    return np.clip(O_k, 0, 255).astype(np.uint8)

# Reading input
input_filename = str(input()).rstrip()
method = int(input())

# Opening image
I = imageio.imread(input_filename)

# Method 1: Constrained Least Squares
if method == 1:

    # Reading input
    k = int(input())
    sigma = float(input())
    gamma = float(input())
    
    # Applying Gaussian filter
    h = gaussian_filter(k, sigma)

    width_ax0 = int( (I.shape[0]-h.shape[0]) / 2 )
    width_ax1 = int( (I.shape[1]-h.shape[1] + 1)/2 ) 
    h_padded = np.pad(h, [width_ax0, width_ax1]) 

    # Restoring image
    I_gauss = convolve(I, h_padded)
    I_new = clsq(I_gauss, h_padded, gamma)

# Method 2: Richardson-Lucy
elif method == 2:

    # Reading input
    angle = float(input())
    n_steps = int(input())

    # Applying PSF
    psf = get_motion_psf(I.shape[0], I.shape[1], angle)

    #I_psf = I
    I_new = rl(I, psf, n_steps)

# Printing RMSE 
print(rmse(I, I_new)) 

plt.subplot(221)
plt.title("Original")
plt.imshow(I,cmap="gray")

if method == 1:
    plt.subplot(222)
    plt.title("Gaussian")
    plt.imshow(I_gauss,cmap="gray")

plt.subplot(223)
plt.title("New")
plt.imshow(I_new,cmap="gray")

plt.show()
