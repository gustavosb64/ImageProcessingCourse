# Image restoration

_Video01: Image formation model and common sources of noice_

### **Enhancement** is different from **restoration**
* Enhancement is subjective
* Restoration uses objective methods
### **Degratations**
* Blur
* Motion blur
* Noise

## **Sources of noise**

#### Photon counting
* Light detection via a sensor is a statistical process, well modeled by a Poisson distribution
* The precision of the measured signal is proportional to the mean of the signal (amount of photons)
* The amount of noise can be approximated by the squared root of the number of photons 
* It is not an additive noise
 
#### Thermal (? or still photon counting?)
* Working with extreme focal distance:
    * Smaller pixels capture better fine details
    * Each pixel will have a lower amount of photons
    * Sharper image, but still **noisier**
* Smaller pixels allow to observe more details, but with the cost of a lower **signal-to-noise** ratio.

#### Thermal
* Electrons are generated when a the photons are detected. Those will vary given the temperature of the sensor.
* Usually, we assume this noise to be **Gaussian** and additive, also called **white noise** (when variance = 1).
* Possible way to diminish it: **Dark Frame capture**, an image obtained without light acquisition.
* Dark Frame can then bue subtracted from acquired images
* This image contains a map of the thermal noise.

#### Quantisation
* Caused by quantisation of pixels from continuous to unsigned _int/char_
* It often follows uniform distribution.
* When the quantisation level is low, the noise can becmoe signal dependent and correlated to each region of the image (non-uniform)
 
#### Transmission/display
* Caused by errors in some bits when storing or failure when transmitted.
* Resulting noise is referred to as "impulsive" but also "salt and pepper".
* It affects just a small amount of pixels, but those affected are completely destroyed.
* The mathematical representation of the impulsive noise can be seen as two "impulses" (or _Dirac functions_) in 0 or 255.
    * A random pixel has probability _p_ of being affected by noise, usually _p_/2 for "salt" and _p_/2 for "pepper".
     
_Video02: Noise simulation, filters and adaptive filters_

### Noise generation

### Noise reduction

#### Mean filtering
* **Arithmetic**: increase blur by creating a new value based on the average neighbour pixels.
* **Geometric**: preserve details when pixel differences are in the order of multiples of a given base (ex: it is logarithmic).
* **Harmonic**: reduce the influence of outliers.
 
#### Order statistic filtering
* **Median**:
    * Widely used in pre-processing.
    * Removes texture, preserve edges.
* **Max**:
    * Can be used to locate bright points in the image
* **Min**:
    * Can be used to locate dark points in the image
* **Mean**:
    * Combines order statistics with mean
    * Usually produces an effect similar to median, but often thickens the borders/edges
     
#### Adaptive filtering
* Take in account local statistics.
* The objective is to allow smoother results mostly in flat regions (with less detail).
* Any filter can be developed in an adaptive fashion:
    * Adaptive noise reduction using mean and local variance
    * Adaptive noise reduction using median and local inter-quartile range (IQR)
     
#### Bilateral filtering
* Noise reduction while preserving edges.

_Video03: Implementing noise simulation and denoising filters_

_Video04: Blur effect and leas squares filters_
* The second effect that oftens degrades images is blur
* Usually called point spread function (PSF) 
