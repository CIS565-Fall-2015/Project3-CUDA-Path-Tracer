CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

siqi Huang Tested on: Windows 7, Inter(R) Core(TM) i7-4870 HQ CPU@ 2.5GHz; GeForce GT 750M(GK107) (Personal Computer)

PART I: Camera Setup
The camera is the first part of the project, when projecting into the scene, we use the color'r rgb value to represent its normal direction, and we get the sample image like this ![](img/sample.png). This means we have the correct camera setup

PART II: Basic Path Tracing
We begin with the diffuse surface intersection. When we hit the light or hit nothing, the ray terminate. Or the color acculumate along the path, the bounce of ray is based on the hemisphere cosine ray function. After this is done, we get image like this ![](img/withoutGitter.png). We can see from the image that image that the back wall have fuse color on the two side because of the color of the side walls.

PART III: Perfect Specular Object
The perfect specular object follow the mirror reflection rule If we hit a perfect specular object, we accumate its color and directly rebounce the ray. ![](img/basic1.png)

PART IV: Transparent Object
Using Snell's law we can get the refraction ray. However, in the actual situation, some ray in fact bounce like the perfect specular ray. To simulate this, we use Fresenl's law and in practice we use schlick's approach for simulation. One thing to draw here is that we can not use recursive method to follow all ray, so whether a ray actually rebounce or refract depends on the random number generator. If the num of iteration is larget, the overall effect is correct![](img/ball1.png)

PART V: Mesh Intersection
We first read obj file into our program and copy them to GPU. Here we intersect the ray with each triangle of the object. The copy to GPU may become a problem. In general, first copy all vertex,normal,index of the mesh, then copy the mesh itself. That can guarantee you get the whole structure of the mesh and they are all linked to each other on GPU. ![](img/bunny1.png)

PART VI: 


