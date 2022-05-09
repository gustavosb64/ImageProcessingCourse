import numpy as np
import matplotlib.pyplot as plt
import imageio

def normalisation(g):
    min_value = np.min(g)
    max_value = np.max(g)

    res = (g - min_value) / (max_value - min_value) 
    return res*255

input_filename = str(input()).rstrip()
filter_filename = str(input()).rstrip()
gRef_filename = str(input()).rstrip()

I_input = imageio.imread(input_filename)
I_filter = imageio.imread(filter_filename)
I_gRef = imageio.imread(gRef_filename)

I_test = np.fft.fft2(I_input)
print(I_test)

plt.subplot(221)
plt.imshow(I_input,cmap="gray")
plt.subplot(222)
plt.imshow(I_filter,cmap="gray")
plt.subplot(223)
plt.imshow(I_gRef,cmap="gray")
#plt.subplot(224)
#plt.imshow(I_test,cmap="gray")
plt.show()
