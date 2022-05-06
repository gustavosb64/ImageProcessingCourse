### IMAGE FILTERING (video 3)

    * **Smoothing (variance reduction)**
        * _Mean_: all values in the weight matrix are the mean value
        * _Weighed mean_: like the previous one, but some values can be weighed differently
        * _Gaussian filter 2D_: uses the normal distribution. Assumes that the data distribution are centered around the mean value, so the values closer to the central value has a higher probability. The weight follows this distribution.
            * $\sigma$ (standard variation) defines how it spreads as we get further from the mean value. If we get a _lower_ $\sigma$, the values would have less values deviating from the mean.
        
### IMAGE FILTERING (video 3)
