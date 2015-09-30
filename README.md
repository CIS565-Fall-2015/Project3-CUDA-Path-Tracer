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

PART VI:  Kdtree Acceleration
The above mesh intersection is the naive method, which is OK for small objects, but impossible for larget objects. So we use a kdtree to accelerate the computation. We first create the kdtree on CPU recursively, then copy the structure onto GPU. This process is much more complex than copy a mesh. The copy here is recursively, using a postorder traverse of the tree to help. First copy the left and right child of the tree, pass its data to GPU, link the pointer to the CPU pointer, then copy the whole tree to GPU. The recursive funciton should return the GPU pointer. When all this is done, the acutal simplifaction of intersection is done on GPU. Because we do not know beforehand how much triangle will a ray intersect, I can only hard code the number used for the actual triangle intersection. If we do that on CPU, we can use vector to store the information which means we do not need to deal with the space issue, but on GPU it is different. Kdtree On GPU has many SIGGRAPH paper to reference, to improve this part, a lot of work is needed.
![](img/dragon1.png)![](img/dragon3.png)

PART VII: Stream Compaction
In the interseciton some ray may terminated after some iteration. But we are still running them on GPU. So if we can reduce the number of block used for this, we can greatly improve the performance here. Here I used the shared memory with the efficient scan(upswap and downswap) to deal with the scan in one block. And among block I add the last number of each block, scan them again and add them back to the original global data. After the stream compaction is accomplished, if the depth is very large, the time used will reduce dramatically when compared method without stream compaction.

PART VIII: Gitter Within Ray:
To solve the problem of aliasing, we used a gittered ray for each pixes in each iteration.
There are a comparison of images:
![](img/withoutGitter.png)
![](img/withGitter.png)
The first is without the gitter, the sphere boundary is "hard" and the ceilling has a very clear line boundary with the left and right wall, but the second does not.

PART IX: Performance Analysis
  1 stream compaction vs no stream compaction:
    The first time I did this I put the depth in the thread, and it is not very slow. When implementing the stream compaction and the depth is low, the speed may be slower than not do that. However, the time used with respect to the depth is linear in this case. But after implementation the stream compaction, this relationship becomes Logarithmic. The following graph show's the number of thread after stream compaction with respect to the depth. ![](img/iter-thread.png).
    When the depth is very large, the advantage of the stream compaction is very obvious
  2 Frensel's law vs no Frensel's law
    I believe this does not affect a lot in the performance. If using the Frensel's law, in each iteration I only have on more random number generation on device. Becuase the ray is either reflected or refracted, either way we compute just on block in the function but not both.
  3 Kdtree vs no Kdtree
    Using the naive method, one iteration with a sphere object with 2000+ faces need almost 10s. However, if using the kdtree structure, one iteration use less than one third second. Theoretically, the time complexity of the kdtree is also logrithmic, and the naive method is also linear. However, in runtime the kdtree may not be that fast due to the space need to store all the triangle indexs. The space needed maybe large(say rendering of the stanford dragon with 100k faces), which may reduce the performance of a block.
  4 open scene vs closed scene
    I actually not tested on the closed scene, but I am sure it will be slower than the open scene. Because in the open scene a lot of ray terminated by hitting nothing, but in the closed scene that never happens. This effect will be enlarged when the depth is very large.
  5 global variable vs loacl variable
    In the beginning I used a lot of local variables. I initiate them, allocate them and free them in every iteration, even in every depth. Then I changed the local variables to global variables and I only need to allocate and free once. And the fps for a simple scene raised from 5 to 6.6. That means the cuda malloc and memcpy may use a lot of time.
