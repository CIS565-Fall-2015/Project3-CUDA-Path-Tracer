CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ziwei Zong
* Tested on: Windows 10, i7-5500 @ 2.40GHz 8GB, GTX 950M (Personal)

Description
========================
--------------------------
## Overview

This GPU based path tracer with global illumination and anti-alising can render diffuse, perfect/non-perfect specular, transparent and subsurface scattering materials. Shown as the picture below.

![](img/01Overview.png)
--------------------------
## Features

### Materials

Below are material types that this path tracer supports:
 * **Diffuse**
 * **Specular**
     ** Perfect Specular (Mirrors)
     ** Non-perfect Reflection
 * **Transparent (with fresnel reflection)**
 * **Subsurface Scattering**
     ** Diffuse Subsurface
     ** Subsurface with reflection
 ![alt tag](img/04DiffSpecTrans.png "scene file: \scenes\DiffSpecTrans.txt")
 
#### Subsurface Scattering
![](img/05SSS.png)
Subsurface scattering is implemented based on Yining Karl Li's [Slides](https://github.com/CIS565-Fall-2015/cis565-fall-2015.github.io/raw/master/lectures/4.1-Path-Tracing-1.pdf)
 Subsurface Scattering	|  Compare with diffuse
:----------------------:|:-------------------------:
![](img/05SSS02.png)		|![](img/05SSS01.png)

### Texture Mapping
![](img/06TexMap.png)
Cube Texture Mapping Test |Sphere Texture Mapping test
:------------------------:|:---------------------------:
![](img/06TexMap_cube.png)|![](img/06TexMap_sphere.png)

### Global Illumination

With Direct Lighting	|  Without Direct Lighting
:----------------------:|:-------------------------:
![](img/02Gobal_on.png)|![](img/02Gobal_off.png)

#### Anti-aliasing

Choose a random direction inside each pixel to smooth edges while iterating.
With Anti-Aliasing		|Without Anti-Aliasing
:----------------------:|:------------------:
![](img/03AntiA_on.PNG) |![](img/03AntiA_off.png)

### Work-efficient Stream Compaction with Shared Memory
![](img/Analysis/SharedMem.PNG)

--------------------------
## Analysis
**Open vs Closed scenes**

![](img/Analysis/OpenScene.png)
![](img/Analysis/CloseScene.png)

--------------------------
## Appendix
#### command line
#### control
#### scene file
#### base code
#### future work
#### references
