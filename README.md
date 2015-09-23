CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

Instructions (delete me)
========================

#### Controls

* Esc to save an image and exit.
* Space to save an image. Watch the console for the output filename.
* W/A/S/D and R/F move the camera. Arrow keys rotate.

## Requirements

You will need to implement the following features:

* ~~Raycasting from the camera into the scene through an imaginary grid of pixels~~
  (the screen)
  * ~~Implement antialiasing (by jittering rays within each pixel)~~
* ~~Diffuse surfaces~~
* ~~Perfectly specular-reflective (mirrored) surfaces~~
  * ~~See notes on diffuse/specular in `scatterRay` and on specular below~~
* **NEWLY ADDED:** Work-efficient stream compaction using shared memory across
  multiple blocks (See *GPU Gems 3* Chapter 39).

* Features:
  * Refraction (e.g. glass/water) with Frensel effects using Schlick's
    approximation
  * Physically-based depth-of-field (by jittering rays within an aperture)
  * Recommended but not required: non-perfect specular surfaces
  * Subsurface scattering
    * SSS for reflective material

For each extra feature, you must provide the following analysis:

* Overview write-up of the feature
* Performance impact of the feature
* If you did something to accelerate the feature, what did you do and why?
* Compare your GPU version of the feature to a HYPOTHETICAL CPU version
  (you don't have to implement it!) Does it benefit or suffer from being
  implemented on the GPU?
* How might this feature be optimized beyond your current implementation?

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
* EYE (float x) (float y) (float z) //camera's position in worldspace
* VIEW (float x) (float y) (float z) //camera's view direction
* UP (float x) (float y) (float z) //camera's up vector

Objects are defined in the following fashion:

* OBJECT (object ID) //object header
* (cube OR sphere OR mesh) //type of object, can be either "cube", "sphere", or
  "mesh". Note that cubes and spheres are unit sized and centered at the
  origin.
* material (material ID) //material to assign this object
* TRANS (float transx) (float transy) (float transz) //translation
* ROTAT (float rotationx) (float rotationy) (float rotationz) //rotation
* SCALE (float scalex) (float scaley) (float scalez) //scale

Two examples are provided in the `scenes/` directory: a single emissive sphere,
and a simple cornell box made using cubes for walls and lights and a sphere in
the middle.

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
