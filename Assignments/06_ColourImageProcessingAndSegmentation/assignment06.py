# Student: Gustavo 
# USP number: 
# Course code: SCC0251
# Year/semestre: 2022/1st semester
# Title: Assignment06 - Color Image Processing and Segmentation

import numpy as np
import imageio
import random

# Normalize the image between new_min and new_max
def normalization(g, new_min, new_max):
    min_value = np.min(g)
    max_value = np.max(g)

    res = ( (g - min_value) * ( ( new_max - new_min ) / (max_value - min_value) ) ) + new_min

    return res

# Returns the RMSE between original image I and the new image I_new
def rmse(I, I_new):
    return np.sqrt(np.sum((I - I_new)**2)/(I.shape[0] * I.shape[1]))

# Returns the resulting images after segmentation
def resulting_image(I, op_attributes, C, groups):

    # If it uses RGB
    if op_attributes == 1 or op_attributes == 2: 
        I_new = np.zeros(I.shape)

        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                for k in range(3):
                    I_new[i, j, k] = C[groups[i, j]][k]

    # If it uses Luminance
    else:
        I_new = np.zeros(I.shape)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                I_new[i, j] = C[groups[i, j]] 

    return I_new
    
# K-Means method
def k_means(I, I_flat, k, n, C, groups, n_attr, op_attributes):

    # Calculates the distance between each cluster and pixel
    dist_list = np.zeros([I.shape[0]*I.shape[1],0]) 
    for centroid in C:
        c_pos  = np.full((I.shape[0]*I.shape[1], n_attr), centroid)

        distances = np.sqrt(np.sum((I_flat-c_pos)**2, axis=1))
        distances = np.reshape(distances,[I.shape[0]*I.shape[1], 1])

        dist_list = np.append(dist_list,distances,axis=1)

    # The minimum value in dist_list is the cluster where the element belongs
    groups = np.argmin(dist_list, axis=1)

    # Update centroids
    for i, centroid in enumerate(C):
        centroid = np.mean(I_flat[np.where(groups==i)])
        C[i] = centroid
    
    # In case the method has reached the number of iterations, return the resulting image
    if (n == 0):
        groups = np.reshape(groups, [I.shape[0],I.shape[1]]) 
        return resulting_image(I, op_attributes, C, groups)
    else:
        return k_means(I, I_flat, k, n-1, C, groups, n_attr, op_attributes)

# Reading input
input_filename = str(input()).rstrip()
ref_filename = str(input()).rstrip()
op_attributes = int(input())    #option for pixel attributes
k = int(input())    # number of clusters
n = int(input())    # number of iterations
seed = int(input())

# Opening images
I = imageio.imread(input_filename).astype(np.float32)
I_ref = imageio.imread(ref_filename).astype(np.uint8)

print(I.shape)

# Generating centroids indices
random.seed(seed)
idx_centroids = np.sort(random.sample(range(0, I.shape[0]*I.shape[1]), k))

# RGB
if op_attributes == 1:
    I_flat = np.reshape(I, [I.shape[0]*I.shape[1], 3])

    # Initialise centroids in a matrix
    C = np.zeros( (k, 3) )
    for i in range(k):
        C[i] = I_flat[idx_centroids[i]]

# RGBxy
elif op_attributes == 2: 

    X = np.tile(np.reshape(np.arange(I.shape[0]), [I.shape[0],1]), (1,I.shape[1]))
    Y = np.tile(np.reshape(np.arange(I.shape[1]), [1,I.shape[1]]), (I.shape[0],1))
    X = np.reshape(X, [I.shape[0]*I.shape[1],1])
    Y = np.reshape(Y, [I.shape[0]*I.shape[1],1])
    XY = np.concatenate((X,Y),axis=1)

    I_flat = np.reshape(I, [I.shape[0]*I.shape[1], 3])
    I_flat = np.concatenate((I_flat,XY),axis=1)

    # Initialise centroids in a matrix
    C = np.zeros( (k, 5) )
    for i in range(k):
        C[i] = I_flat[idx_centroids[i]]

# Luminance
elif op_attributes == 3:                                    

    I_flat = np.reshape((0.299*I[:,:,0]) + (0.587*I[:,:,1]) + (0.114*I[:,:,2]), [I.shape[0]*I.shape[1],1])

    # Initialise centroids in a matrix
    C = np.zeros( (k) )
    for i in range(k):
        C[i] = I_flat[idx_centroids[i]]

# Luminance xy
elif op_attributes == 4:

    X = np.tile(np.reshape(np.arange(I.shape[0]), [I.shape[0],1]), (1,I.shape[1]))
    Y = np.tile(np.reshape(np.arange(I.shape[1]), [1,I.shape[1]]), (I.shape[0],1))
    X = np.reshape(X, [I.shape[0]*I.shape[1],1])
    Y = np.reshape(Y, [I.shape[0]*I.shape[1],1])
    XY = np.concatenate((X,Y),axis=1)

    I_flat = np.reshape((0.299*I[:,:,0]) + (0.587*I[:,:,1]) + (0.114*I[:,:,2]), [I.shape[0]*I.shape[1],1])
    I_flat = np.concatenate((I_flat,XY),axis=1)

    # Initialise centroids in a matrix
    C = np.zeros( (k, 3) )
    for i in range(k):
        C[i] = I_flat[idx_centroids[i]]

# Creating clusters 
groups = np.zeros(I.shape[0]*I.shape[1]).astype(np.float32)
n_attr = I_flat.shape[1]

# Using K-Means method
I_new = k_means(I, I_flat, k, n, C, groups, n_attr, op_attributes)

# Applying RMSE 
if (op_attributes == 1 or op_attributes == 2):

    rmse_all_colours = np.zeros(3)

    # Normalising for each colour channel
    for i in range(3):
        I_new[:,:,i] = normalization(I_new[:,:,i], 0, 255)
        rmse_all_colours[i] = rmse(I_ref[:,:,i], I_new[:,:,i])

    rmse_res = sum(rmse_all_colours)/3

else:
    I_new = normalization(I_new,0,255) 
    rmse_res = rmse(I_new, I)

print(rmse_res)
