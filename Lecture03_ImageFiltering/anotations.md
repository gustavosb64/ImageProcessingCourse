### IMAGE FILTERING 

* **Smoothing (variance reduction)** (video 3)
    * _Mean_: all values in the weight matrix are the mean value
    * _Weighed mean_: like the previous one, but some values can be weighed differently
    * _Gaussian filter 2D_: uses the normal distribution. Assumes that the data distribution are centered around the mean value, so the values closer to the central value has a higher probability. The weight follows this distribution.
        * $\sigma$ (standard variation) defines how it spreads as we get further from the mean value. If we get a _lower_ $\sigma$, the values would have less values deviating from the mean.
        
* **Differencial filters** (video 4)
    * Computes transitions on intensities.
    * They are used to enhance details and to increase the noise in an image because it is very sensitive to any variation on the image.
    * _Laplacian filter_
     
* **Sharpening filters** (video 4)
    * Enhances details (and noise)
    * _Laplacian sharpening_
    * _Unsharp mask_
