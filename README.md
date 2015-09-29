CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ziwei Zong
* Tested on: Windows 10, i7-5500 @ 2.40GHz 8GB, GTX 950M (Personal)

Description
========================

## Overview
This GPU based path tracer with global illumination and anti-alising can render diffuse, perfect/non-perfect specular, transparent and subsurface scattering materials. Shown as the picture below.

![](img/01Overview.png)

## Features

### Materials

 * **Diffuse**
 * **Perfect Specular Materials (Mirrors)**
 * **Specular**
 * **Transparent (with fresnel reflection)**
 * **Subsurface Scattering**
 ![alt tag](img/04DiffSpecTrans.png "scene file: \scenes\DiffSpecTrans.txt")
 
## Subsurface Scattering
 ![](img/05SSS.png)

 Subsurface Scattering	|  Compare with diffuse
:----------------------:|:-------------------------:
![](img/05SSS02.png)		|![](img/05SSS01.png)

### Global Illumination

With Direct Lighting	|  Without Direct Lighting
:----------------------:|:-------------------------:
![](img/02Gobal_on.png)|![](img/02Gobal_off.png)

#### Anti-aliasing

With Anti-Aliasing		|Without Anti-Aliasing
:----------------------:|:------------------:
![](img/03AntiA_on.PNG) |![](img/03AntiA_off.png)

### Work-efficient Stream Compaction with Shared Memory


## Analysis
   #### stream compaction
        (shared mem work-eff/thrust::remove_if/no stream compact)
   #### Open vs Closed scenes

## Appendix

#### command line
#### control
#### scene file
   #### future work
   #### base code
   #### references
