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

## Requirements

* Features:
  * Work-efficient stream compaction using shared memory
  * Raycasting from the camera into the scene through an imaginary grid of pixels
    * Casting to plane at distance based on FOV
  * Diffuse surfaces
    * Cosine weighted
  * (E) Non-perfect specular surfaces
    * Cosine weighted, restricted by specular exponent
    * Perfectly specular-reflective (mirrored) surfaces would be a non-perfect surface with very large (toward positive infinity) specular exponent
    * http://www.cs.cornell.edu/courses/cs4620/2012fa/lectures/37raytracing.pdf
  * (E) Refraction (ice / diamond) with Frensel effects using Schlick's approximation
    * https://en.wikipedia.org/wiki/Schlick's_approximation
  * (E) Subsurface scattering
    * Simplified version of Dipole
      * https://graphics.stanford.edu/papers/bssrdf/bssrdf.pdf
      * Reduced memory overhead by approximating ray-out position without passing geometry all the way into scatter function
      * Passing geometry slows the entire rendering process by a factor of ~3
    * SSS for reflective material (split on specular and diffuse, then further split on diffuse)
  * Antialiasing
    * Oversampling at each iteration
    * Effectively increases render time; proportional to # of oversampling passes
  * (E) Physically-based depth-of-field (by jittering rays within an aperture)
    * Using antialiasing routines but with different jittering method
    * Find focal plane
    * Jitter each ray on its origin
    * Keep ray end point intact so it always focuses on focal plane; equivalent to jittering camera itself around focal plane

* Scenes:
  * `cornell1`: mixed objects (specular, refraction, diffuse, caustic)
  * `cornell2`: SSS, same size spheres, mixed distances to light
  * `cornell3`: SSS + specular
  * `cornell4`: SSS, same size cubes, mixed distances to light
  * `cornell5`: mixed objects, closed scene, camera inside box
  * `cornell6`: single sphere, for performance testing

For each extra feature, you must provide the following analysis:

* Overview write-up of the feature
* Performance impact of the feature
* If you did something to accelerate the feature, what did you do and why?
* Compare your GPU version of the feature to a HYPOTHETICAL CPU version
  (you don't have to implement it!) Does it benefit or suffer from being
  implemented on the GPU?
* How might this feature be optimized beyond your current implementation?

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

## Performance
* Baseline: `cornell6`, 200*200, scan block size 64
* Scan:
  * Occupacy with block size
    * Block size needs to be 2^n: pick 128, occupacy 33.3% -> 41.28%
  * Occupacy with register counts
    * Wasn't able to reduce register count, but reduced execution time
      * Remove shorthand variable for `threadIdx.x`: no effect (should have used one less register)
      * Store `threadIdx-1` to avoid repetitive calculations: no effect on registers, but reduced execution time from ~7700 to ~4000 microsec
      * Change loop ceiling `ilog2ceil(n)` to a compile-time constant: no effect, since only two less calculations per thread
      * Pre-populate `2^d` values into an array in shared memory: no effect on registers, but reduced execution time from ~4000 to ~330 microsec
    * Conclusion: achieved occupacy 41.28% -> 40.6%; exec time: ~7700 -> ~330 microsec; 2300% speed up
* Scan scatter (stream compaction):
  * Pre-cache data in shared memory
    * Reduced global memory replay overhead 50% -> 24.3%
    * L1 cache 1% -> 86.7%
    * Slight exec time speed up on larger array
    * Trade-off:
      * Reduced occupacy due to increased register count: 7 -> 23, 66.6% -> 33.3%
      * Minimal exec time loss on small arrays ~200 microsec
* Intersection test:
  * Block size left unchanged as optimal
  * Pre-cache geometries in shared memory
    * Reduced access to device memory by 30% in data size, without obvious change in exec time
      * More access to L1 cache as a result
      * In some cases exec becomes faster but not consistent
    * LIMITATION: Current code allows only 64 geometries can be efficiently loaded, which is equal to the block size; this is fine for non-mesh geometries and less than 64 geometries. For bigger scenes, extra codes for loading more geometries will be needed for it to work
  * Move temporary variable declarations out of for-loop; change all parameters of intersection test to pass by reference
    * On top of above memory improvement, further reduced access to device memory by 30% in data size
    * However, this only reduces access to thread's temporary variables. Therefore the reduction might not scale with scene window size
    * 14.5% speed up on exec time (reduce by ~200 microsec)
    * The speed up and memory improvement are cancelled if the methods called by intersection tests also have all their parameters passed by reference
  * Remove temporary variable that stores `(ray, color)` pair, which is stored as a struct
    * Contrary to the belief that the temporary variable is caching and is faster, direct access is actually better
    * Further reduced device memory access by 40%
    * Further reduced exec time by ~150 microsec
    * Reduced register count by 1
    * Increased access to L1 cache: 65% -> 85%
    * Increased global memory replay overhead: 25% -> 34.7%
* Ray scatter:
  * Remove temporary variable that stores `(ray, color)` pair, which is stored as a struct
    * Reduced register count: 63 -> 55
    * 63% exec time speed up
    * 50% less device memory access (data size)
    * L2 cache 31.6% -> 40.6%
    * Global memory replay overhead 49.2% -> 39.3%
  * Refactor `scatterRay` to remove redundant parameters
    * Minor improvements regarding items above
* Path termination
  * Remove temporary variable for `(ray, color)` pair and material
    * Reduced register count: 29 -> 12
    * 586% exec time speed up
    * Trade-off: high global memory replay overhead (47.5%)
* Camera raycasting
  * Remove temporary variable for `(ray, color)` pair
    * 49.5% speed up

### Analysis

* Creating a temporary variable for caching elements in large arrays is not always effective in CUDA kernels. In fact, directly passing the array element around results in much better performance in terms of both exec time and memory access. For example, caching a `(ray, color)` pair for later computation only reduces performance in long computations such as intersection test. On the other hand, caching geometries for repetitive access across different threads increases performance. However, a caching variable increases performance in simple kernels like pixel painting.

* Stream compaction helps most after a few bounces. Print and plot the
  effects of stream compaction within a single iteration (i.e. the number of
  unterminated rays after each bounce) and evaluate the benefits you get from
  stream compaction.
```
800 * 800 cornell1
Depth: 0 / Grid size: 625125
Depth: 1 / Grid size: 497276
Depth: 2 / Grid size: 406024
Depth: 3 / Grid size: 326613
Depth: 4 / Grid size: 264707
Depth: 5 / Grid size: 214660
Depth: 6 / Grid size: 174592
Depth: 7 / Grid size: 141986
```

* Compare scenes which are open (like the given cornell box) and closed
  (i.e. no light can escape the scene). Again, compare the performance effects
  of stream compaction! Remember, stream compaction only affects rays which
  terminate, so what might you expect?
  * Open scene much faster. More rays got terminated due to rays shooting side ways are off to the ambient and thus no hits
  * Closed scene much slower. Less rays got terminated because all rays will hit at least a wall; theoretical 3x more slower with 3 passes; about 2x slower perceived. Closed scene is much brighter.

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
