CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Levi Cai
* Tested on: Windows 8, i7-5500U @ 2.4GHz 12GB RAM, NVIDIA GeForce GTX 940M 2GB

This is a path tracer implemented in CUDA that supports basic ray casting, diffuse and specular surfaces, sphere and cube intersections, work efficient-stream compaction with shared memory, motion blurring, refraction, and depth-of-field.

### Examples of Features Implemented

## Specular vs. Diffuse Surfaces

![](img/cornell_specular.png)

## Motion Blurring

![](img/cornell_bounce.png)

## Refraction (Glass)

![](img/cornell_glass.png)

## Depth of Field

![](img/cornell_dof.png)

### Analysis of Work-Efficient Shared Memory Stream Compaction

I implemented the work-efficient stream compaction with shared memory scan function as described by http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

This allows for significant performance gains in OPEN environments where rays are quickly terminated (or in environments with an extreme number of lights). We can see some of the effects in the graphs below:

![](img/cornell_alive_vs_depth.png)

(Above) I compared the depth in the original cornell image with the number of still-living rays. There is a logarithmic drop-off in the scene of number of still living rays that must be tracked, but for the first several depths the gains are quite dramatic as rays leave the open side of the box quickly.

![](img/cornell_initial_vs_time.png)

(Above) Here we wish to see the effect of very open environments vs. the runtime. To see this, I simply moved the camera's starting position back in the original Cornell image setup (so many rays immediately terminate) and we can see that the performance increases linearly with the number of rays that immediately terminate. We can thus imagine a completely enclosed scene to gain nothing from stream compaction (if anything, it hurts a small amount to do the extra computation due to the cudaMallocs and such, though many of those could possibly be avoided).

