CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Bradley Crusco
* Tested on: Windows 10, i7-3770K @ 3.50GHz 16GB, 2 x GTX 980 4096MB (Personal Computer)

## Description
An interactive GPU accelerated path tracer with support for diffuse, specular, mirrored, and refractive surfaces. Additional features include depth of field and motion blur effects.
![](img/cornell_main_20k.png "Cornell Box")

## Features

### Raycasting???

### Diffuse Surfaces
Diffuse surfaces are supported using a cosine weighted random direction calculation.

### Perfectly Specular Reflective Surfaces
Perfectly specular surfaces give a mirrored effect and are created by combining a specular light component with the calculation of the direction of a ray off a mirrored object.

### Work Efficient Stream Compaction



### Depth of Field

### Non-Perfect Specular Surfaces
![](img/cornell_np_spec.2015-09-26_23-41-17z.5000samp.png "Non-Perfect Specular Surface")

* **Overview**: Non-perfect specular surfaces, which gives a glossy effect, are created using a probability distribution between the diffuse and specular component of a material. First a probability of either a diffuse or a specular ray bounce occuring is calculating by weighting the intensity of the diffuse and specular color values respectively. A random value between 0 and 1 is then generated, which I use to choose a bounce type. The corresponding ray bounce direction is then calculated, as is the color, which is the given color provided by the scene file multiplied by the inverse probability that this bounce occured.
* **Performance Impact**: Neglegable. The only additional calculation to be done is the calculation of the ratio between both color intensities. There is a conditional, which may have performance impact, but this method only calculates one color and one ray bounce just like the mirrored and diffuse implementations.
* **GPU vs. CPU Implementation**: A CPU implementation would likely be recursive, where my GPU implementation is not. Because of this I use a probability calculation to determine how to bounce and only do the bounce once. Since the CPU implementation is recursive, it would likely trace both the specular and diffuse bounces instead of just picking one, and then use the ratio to determine the weights of the resulting color. So for the CPU implemenation I would expect dramatically more performance requirements for this feature than my GPU implemenation.

### Refractive Surfaces with Fresnel Effects
* **Overview**: This is calculated in much the same way as non-perfect specular surfaces. We figure out a probability that a ray hitting our refractive surface will either bounce off and reflect or pass into and refract through the object. If it reflects, we calculate the mirrored reflection direction, and if it refracts we calculate the ray direction using [Snell's law](https://en.wikipedia.org/wiki/Snell%27s_law). The main difference is in the calculation of this probability. We calculate the Fresnel reflection coefficient using [Schlick's approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation) (the inverse of which is the refraction coefficient). An index of refraction, specified in the scene files, determines the refractive properties of the respective material. Air has an index of refraction of 1, and glass about 2.2. It is important to keep track of whether a ray is going into an object or coming out of it, as the indexes are used in a ratio, and the ordering changes depending on what is being exited and what is being entered.
* **Performance Impact**: Significant. Compared to non-perfect specular surfaces, we have many more calculations to do to figure out the respective reflection and refraction coefficients. In addition, if a ray hits a refractive object at a perpendicular angle, the ray is always reflected, regardless of our reflection and refraction coefficients. This is another additional calculation that adds to the performance demands.
* **GPU vs. CPU Implementation**: As far as comparing my GPU implementation to what I'd expect a CPU implementation to be, it'd be the same as the comparison for non-perfect specular surfaces, except in this case the performance increase would be much more significant because it would have to make the additional calculations for the Fresnel coefficients.
* **How to Optimize**: To keep track of whether a ray was inside or outside of an object so I could know how to use the index of refraction coefficients (the air and the other object) I added a boolean to my Ray struct that held this state. This significantly added to my memory overhead because I'm using so many Ray's. I'd like to come up with a way to determine this property on the fly without saving it to further optimize performance.

### Motion Blur
* **Overview**: Motion blur is very conceptually simple. We merely transform an object's position over the course of our render. This creates a blur effect because we are sampling the object at different locations, which creates a blurry trail as the object moves across the screen. 
* **Performance Impact**: 
* **GPU vs. CPU Implementation**: 
* **How to Optimize**: 


## Analysis
### Stream Compaction: Open vs. Closed Scenes

#### Active Threads Remaining at Each Trace Depth

![](img/Project 3 Analysis 1.png "Active Threads Remaining at Trace Depth (Open vs. Closed Scene")

#### Trace Execution Time at Each Trace Depth

![](img/Project 3 Analysis 2.png "Trace Execution Time at Trace Depth (Open vs. Closed Scene")

### Stream Compaction: Compaction vs. No Compaction

![](img/Project 3 Analysis 3.png "Trace Execution Time at Trace Depth for an Open Scene (Compaction vs. No Compaction")

![](img/Project 3 Analysis 4.png "Trace Execution Time at Trace Depth for a Close Scene (Compaction vs. No Compaction")

## Interactive Controls

## Scene File Format 
Add something on included scenes
