# Student: Gustavo 
# USP number: 
# Course code: SCC0251
# Year/semestre: 2022/1st semester
# Title: Assignment03 - Filtering in Spatial and Frequency Domain

import numpy as np
import matplotlib.pyplot as plt
import imageio

# Normalize the image between new_min and new_max
def normalization(g, new_min, new_max):
    min_value = np.min(g)
    max_value = np.max(g)

    res = ( (g - min_value) * ( ( new_max - new_min ) / (max_value - min_value) ) ) + new_min

    return res

# Returns the RMSE between original image I and the new image I_new
def rmse(I, I_new):
    return np.sqrt(np.sum((I - I_new)**2)/(I.shape[0] * I.shape[1]))

# Reading input
input_filename = str(input()).rstrip()
filter_filename = str(input()).rstrip()
ref_filename = str(input()).rstrip()

# Opening images
I_input = imageio.imread(input_filename)
I_filter = imageio.imread(filter_filename)
I_ref = imageio.imread(ref_filename)

# Normalizing filter between 0 and 1
I_filter = normalization(I_filter, 0, 1)

# Generating Fourier Spectrum
I_fft = np.fft.fft2(I_input)

# Shifting the spectrum so the zero-frequency component go to the center 
I_fft = np.fft.fftshift(I_fft)

# Multiplying the Fourier Spectrum by the filter
I_new = np.multiply(I_fft, I_filter)

# Inverting the shift
I_new = np.fft.ifftshift(I_new)

# New plot uses the real value of the image
I_new = np.real(np.fft.ifft2(I_new))
I_new = normalization(I_new,0,255).astype(np.uint8)

# Printing RMSE
print(rmse(I_ref, I_new))

# Ploting images
I_fft = np.log(np.abs(I_fft)+1); 

plt.subplot(321)
plt.title("Original")
plt.imshow(I_input,cmap="gray")
plt.subplot(322)
plt.title("My filtered image")
plt.imshow(I_new,cmap="gray")
plt.subplot(323)
plt.title("Spectrum")
plt.imshow(I_fft,cmap="gray")
plt.subplot(324)
plt.title("Reference")
plt.imshow(I_ref,cmap="gray")
plt.subplot(325)
plt.title("Filter")
plt.imshow(I_filter,cmap="gray")
plt.show()
