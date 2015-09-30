CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Kangning Li
* Tested on: Windows 10, i7-4790 @ 3.60GHz 16GB, GTX 970 4096MB (Personal)

![](images/toBoldlyGo.gif)

This repository contains a CUDA path tracer with the following features:
- basic diffuse and perfectly reflective materials
- work-efficient shared memory stream compaction
- obj file loading and naive rendering
- motion blur

This was done as HW3 for CIS 565 2015, GPU Programming at the University of Pennsylvania.

**Materials**
![](images/cornell.2015-09-28_12-01-33z.1409samp.png)

The content of this assignment was weighted more towards learning CUDA than pathtracing. As such, this path tracer only supports perfectly reflective and perfect "diffuse" materials at the moment. Code in interactions.h exists with the intent to enable support in the future for simple refraction as well as fresnel reflection/refraction approximated using Schlick's law. However, these features are not complete as can be seen below:

![](images/cornell.2015-09-29_06-25-17z.1403samp.png)

We can still get some interesting images, however:

![](images/cornell.2015-09-29_12-26-54z.5000samp.png)

Note that all lights are just geometry objects with an emissive material.