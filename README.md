CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ziye Zhou
* Tested on: Windows 8.1, i7-4910 @ 2.90GHz 32GB, GTX 880M 8192MB (Alienware)

Teaser Image
========================
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/teaser.png?raw=true)

Progress
========================
### Casting Ray From Camera

In order to make sure that the ray casting from camera to the scene is right, I visualize the direction of the casting ray to debug this part.
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-24_03-02-57z.60samp.png?raw=true)

### Getting the Basic to Work
When finish implementing the interative path tracing algorithm and light scattering of diffuse material, I got something as below.
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-24_04-14-01z.138samp.png?raw=true)

This is really wierd since everything seems to work fine except the back wall. Then I figure out that it may be caused by numerical error since we are given realy thin wall. Therefore, I tried to change the thickness of wall to a larrger number and the problem is fixed!
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-24_04-27-43z.2104samp.png?raw=true)

### Specular Material
The ideal specular material is easy to implement, I got something like this:
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-24_23-25-37z.512samp.png?raw=true)

While for the imperfect specular materials, they are simulated using a probability distribution instead computing the strength of a ray bounce based on angles. I used the [Equations 7, 8, and 9 of GPU Gems 3 Chapter 20](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch20.html) to generate a random specular ray. I got something like this:
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-24_23-54-56z.252samp.png?raw=true)

By tweaking the SPECX, we can actually get different effect of the specular material:
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-24_23-55-37z.330samp.png?raw=true)

### Glass (Refractive) Material

First, I implement something according to the [Snell's Law](https://en.wikipedia.org/wiki/Snell%27s_law).
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-25_01-41-58z.2004samp.png?raw=true)

As we can see from the image, we can get the concentration effect for free! Based on that, I also add the [Fresnel Law](https://en.wikipedia.org/wiki/Fresnel_equations) to get more realistic refractive effect.
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-25_17-46-24z.2016samp.png?raw=true)

Compare these two images, we can see that Fresnel Law can provide us with the highlight on the glass material and also some different effect around the edge of the object. 

### Everything Together
Putting everything together, we can get this!
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-30_01-03-22z.2018samp.png?raw=true)

### Arbitrary Mesh

In order to make the scene more interesting, I decided to put the .obj file into the scene. First, I edited my __OBJ Loader from CIS 560__ to work on the GPU. Then, I implemented my own ray triangle intersection test as a `__host__ __device__` function. I also generate some debug image to test the correctness of the intersection function (test both intersect and the normal direction).
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/debug_image.png?raw=true)

After getting all these to work, I start running some test on the stanford bunny with diffuse ans refractive material!
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-27_22-58-21z.2737samp.png?raw=true)
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/cornell.2015-09-29_15-31-08z.2681samp.png?raw=true)

Since brute force intersection test can be the bottle neck when we are importing large mesh, I also implement the __AABB(Axis Aligned Bounding Box)__ for each OBJ so that before testing the ray triangle intersection, I will do the fast Ray Box intersection first to avoid some unnecessary computation. 
 
### Stream Compation with Shared Memory
I am actually testing the new method on top on last project, and I get the testing result as follows:
![alt tag](https://github.com/ziyezhou-Jerry/Project3-CUDA-Path-Tracer/blob/master/img/share_mem_testing.png?raw=true)

It proved our assumption that using shared memory can speed up the program.
