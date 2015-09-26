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
* Overview: Non-perfect specular surfaces, which gives a glossy effect, are created using a probability distribution between the diffuse and specular component of a material. First a probability of either a diffuse or a specular ray bounce occuring is calculating by weighting the intensity of the diffuse and specular color values respectively. A random value between 0 and 1 is then generated, which I use to choose a bounce type. The corresponding ray bounce direction is then calculated, as is the color, which is the given color provided by the scene file multiplied by the inverse probability that this bounce occured.
* Performance Impact: Neglegable. The only additional calculation to be done is the calculation of the ratio between both color intensities. There is a conditional, which may have performance impact, but this method only calculates one color and one ray bounce just like the mirrored and diffuse implementations.
* GPU vs. CPU Implementation: A CPU implementation would likely be recursive, where my GPU implementation is not. Because of this I use a probability calculation to determine how to bounce and only do the bounce once. Since the CPU implementation is recursive, it would likely trace both the specular and diffuse bounces instead of just picking one, and then use the ratio to determine the weights of the resulting color. So for the CPU implemenation I would expect dramatically more performance requirements for this feature than my GPU implemenation.

### Refractive Surfaces


### Motion Blur


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
