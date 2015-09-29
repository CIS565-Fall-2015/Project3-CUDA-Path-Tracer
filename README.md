CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* SANCHIT GARG
* Tested on: Mac OSX 10.10.4, i7 @ 2.4 GHz, GT 650M 1GB (Personal Computer)

### What is a Path Tracer ?

A path tracer is a Global Illumination algorithm for rendering.
The basic setup of the path tracer is that you have a camera, an image to write to and a scene to by rendered. Our aim is to find image pixel colors based on what the camera would be looking at that pixel. To do so, we shoot rays from the camera into the scene, get the intersection points from the scene and based on the material of the intersected object, set the pixel color. This is done over multiple iterations to get a physically correct global illumination. On every iteration, we bounce the ray multiple times based on the material to get a good approximation for a pixel. As every ray is independent of the other, this becomes an embarrassing parallel algorithm.


### Contents

Note: The code is build up on the framework provided by our TA, Kai for the CIS 565 class.

* `src/` C++/CUDA source files.
* `scenes/` Example scene description files.
* `img/` Renders of example scene description files.
* `external/` Includes and static libraries for 3rd party libraries.


#### Controls

* Esc to exit.
* Space to save an image. Watch the console for the output filename.
* W/A/S/D and R/F move the camera. Arrow keys rotate.


## Features

In this project, I aimed at implementing a CUDA base Path Tracer. The features implemented in the project are:

* Ray parallel path tracing over the GPU
* Anti Aliasing
* Basic diffused surface
* Specular surfaces implemented by sampling a light source at random
* Mirror surface
* Refractive surfaces implemented with Fresnels reflectance
* Depth of field
* Direct Illumination by sampling a light source at random
* Attempted Subsurface Scattering
* Work Efficient stream compaction with shared memory to remove dead rays quickly
* Effects like color bleeding and caustics can be observed

### Implementation Details

Let us look at the method used to implement each of the features:

##### Ray parallel path tracing over the GPU ->
As mentioned above, this is an emabarrisingly parallel algorithm as all rays are independent of each other. Hence we can send all the rays in parallel to a CUDA kernel for color calculations.

##### Anti Aliasing ->
A pixel is not a point but a small square area. It is possible that more than one color exists in a pixel and hence the color of the pixel should be the average of all these colors. If we always sample the center of the pixel, we will get the same color and the edges in our final render would be stairstepped. This is called aliasing. To overcome this problem, I jitter the pixel sampling point to select random points within a pixel area. This gives us a better approximation of the color at that pixel and hence a better render. This is called antialiasing.

##### Diffused Surface ->
For a good approximation of the color at a diffused surface, we need to bounce the ray in all possible directions and average the contributions of the color values of all these rays. To get the bounce direction, we take a direction in a hemishpere in the direction of the intersection normal (code provided). This acts as the ray direction for the next iteration.

##### Specular Surface ->
To get a specular highlight, we calculate the color based on the half angle and the specular exponent. To get the half angle, we have to get the ray going to a light. Now as the light is an area light, we sample a random point on the light source to get the light vetor.
Now to decide of the color of the this ray would be a specular or a diffused contribution, we generate a random number between 0 and 1 and take a 30-70 split between the rays. Hence 30% rays contribute specular and 70% contribute diffused. This gives us a desired result.

##### Mirror Surface ->
To get a mirror like effect, we reflect the incoming ray from the surface with respect to the normal.

##### Refractive Surface and Fresnels Reflectance ->
Materials like glass, water etc are refractive surfaces. To implement this, I take the incoming ray and refract it with inside the object. Next, I take this ray and refract it again to go come out. This gives us a transparency like effect. Make sure you feed the correct refractive indices to the refract function.
To make the refraction more physically correct, I implemented Fresnels reflection. As per this law, any incoming light is both refracted and reflected by some amount. The probability of each happening is based on the refractive index of the object. For the calculations I refered to PBRT Page 435. This gave me the probability split between reflection and refraction. The next step is to implement both and add their contributions based on the probability.

##### Depth of Field ->
This is a very interesting effect that can be observed in many photographs where some part of the image is in sharp focus while the other is blurred out. To implement this effect, I used the focal length and aperture parameters of the camera. Assume that there is a sphere centered at the camera position and of the radius of the focal length. We get the intersection of all intitial rays with this sphere. Next, we keep this as the final point but jitter the ray's origin based on the aperture of the camera. The new ray direction will be from this jittered origin to the intersection point. What happens now is that all the points wihtin that focal length are in focus but all others are out of focus.

##### Direct Illumination ->
The concept of direct illumination is that if any ray is alive after the ray depth, then we can take that ray directly to the light. If it can reach the light then we add its contribution to the final image. This helps in generating better looking renders. Also, we can reduce the trace depth and get a similar result with direct illumination.
The important part of this is sampling the lights. For this I borrowed the code from CIS 560 to sample cubes and spheres. First I randomly select a light and then take a random point on that light. This gives me a good sampling of all the light sources.

##### Subsurface Scattering ->
Getting physically accurate subsurface scattering effect is very expensive to compute (Algorithm explained in PBRT). I tried to hack around it and get a very small effect.
In my method, I get a random reflection direction from the intersection point. Next, I shoot a 
----------TODO------------

### Generating random numbers

```
thrust::default_random_engine rng(hash(index));
thrust::uniform_real_distribution<float> u01(0, 1);
float result = u01(rng);
```

There is a convenience function for generating a random engine using a
combination of index, iteration, and depth as the seed:

```
thrust::default_random_engine rng = random_engine(iter, index, depth);
```

### Notes on GLM

This project uses GLM for linear algebra.

On NVIDIA cards pre-Fermi (pre-DX12), you may have issues with mat4-vec4
multiplication. If you have one of these cards, be careful! If you have issues,
you might need to grab `cudamat4` and `multiplyMV` from the
[Fall 2014 project](https://github.com/CIS565-Fall-2014/Project3-Pathtracer).
Let us know if you need to do this.

### Scene File Format

This project uses a custom scene description format. Scene files are flat text
files that describe all geometry, materials, lights, cameras, and render
settings inside of the scene. Items in the format are delimited by new lines,
and comments can be added using C-style `// comments`.

Materials are defined in the following fashion:

* MATERIAL (material ID) //material header
* RGB (float r) (float g) (float b) //diffuse color
* SPECX (float specx) //specular exponent
* SPECRGB (float r) (float g) (float b) //specular color
* REFL (bool refl) //reflectivity flag, 0 for no, 1 for yes
* REFR (bool refr) //refractivity flag, 0 for no, 1 for yes
* REFRIOR (float ior) //index of refraction for Fresnel effects
* SCATTER (float scatter) //scatter flag, 0 for no, 1 for yes
* ABSCOEFF (float r) (float b) (float g) //absorption coefficient for scattering
* RSCTCOEFF (float rsctcoeff) //reduced scattering coefficient
* EMITTANCE (float emittance) //the emittance of the material. Anything >0
  makes the material a light source.

Cameras are defined in the following fashion:

* CAMERA //camera header
* RES (float x) (float y) //resolution
* FOVY (float fovy) //vertical field of view half-angle. the horizonal angle is calculated from this and the reslution
* ITERATIONS (float interations) //how many iterations to refine the image,
  only relevant for supersampled antialiasing, depth of field, area lights, and
  other distributed raytracing applications
* DEPTH (int depth) //maximum depth (number of times the path will bounce)
* FILE (string filename) //file to output render to upon completion
* frame (frame number) //start of a frame
* EYE (float x) (float y) (float z) //camera's position in worldspace
* VIEW (float x) (float y) (float z) //camera's view direction
* UP (float x) (float y) (float z) //camera's up vector

Objects are defined in the following fashion:

* OBJECT (object ID) //object header
* (cube OR sphere OR mesh) //type of object, can be either "cube", "sphere", or
  "mesh". Note that cubes and spheres are unit sized and centered at the
  origin.
* material (material ID) //material to assign this object
* frame (frame number) //start of a frame
* TRANS (float transx) (float transy) (float transz) //translation
* ROTAT (float rotationx) (float rotationy) (float rotationz) //rotation
* SCALE (float scalex) (float scaley) (float scalez) //scale

Two examples are provided in the `scenes/` directory: a single emissive sphere,
and a simple cornell box made using cubes for walls and lights and a sphere in
the middle.

## Third-Party Code Policy

* Use of any third-party code must be approved by asking on our Google Group.
* If it is approved, all students are welcome to use it. Generally, we approve
  use of third-party code that is not a core part of the project. For example,
  for the path tracer, we would approve using a third-party library for loading
  models, but would not approve copying and pasting a CUDA function for doing
  refraction.
* Third-party code **MUST** be credited in README.md.
* Using third-party code without its approval, including using another
  student's code, is an academic integrity violation, and will, at minimum,
  result in you receiving an F for the semester.

## README

Please see: [**TIPS FOR WRITING AN AWESOME README**](https://github.com/pjcozzi/Articles/blob/master/CIS565/GitHubRepo/README.md)

* Sell your project.
* Assume the reader has a little knowledge of path tracing - don't go into
  detail explaining what it is. Focus on your project.
* Don't talk about it like it's an assignment - don't say what is and isn't
  "extra" or "extra credit." Talk about what you accomplished.
* Use this to document what you've done.
* *DO NOT* leave the README to the last minute! It is a crucial part of the
  project, and we will not be able to grade you without a good README.

In addition:

* This is a renderer, so include images that you've made!
* Be sure to back your claims for optimization with numbers and comparisons.
* If you reference any other material, please provide a link to it.
* You wil not be graded on how fast your path tracer runs, but getting close to
  real-time is always nice!
* If you have a fast GPU renderer, it is very good to show case this with a
  video to show interactivity. If you do so, please include a link!

### Analysis

* Stream compaction helps most after a few bounces. Print and plot the
  effects of stream compaction within a single iteration (i.e. the number of
  unterminated rays after each bounce) and evaluate the benefits you get from
  stream compaction.
* Compare scenes which are open (like the given cornell box) and closed
  (i.e. no light can escape the scene). Again, compare the performance effects
  of stream compaction! Remember, stream compaction only affects rays which
  terminate, so what might you expect?


## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project N: PENNKEY`.
   * Direct link to your pull request on GitHub.
   * Estimate the amount of time you spent on the project.
   * If there were any outstanding problems, or if you did any extra
     work, *briefly* explain.
   * Feedback on the project itself, if any.
