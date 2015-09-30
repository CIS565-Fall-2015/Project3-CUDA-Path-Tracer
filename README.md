CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Megan Moore
* Tested on: Windows 7, i7-4770 @ 3.40GHz 16GB (Moore 100 Lab C)

![](img/motion_blur_1000.png "Motion blur after 1000 iterations")

![](img/refraction_correct3(glass).png "Refraction with an IOR of glass")

![](img/final1000.png "Final image after 1000 iterations")

### Analysis

* Stream compaction helps most after a few bounces. Print and plot the
  effects of stream compaction within a single iteration (i.e. the number of
  unterminated rays after each bounce) and evaluate the benefits you get from
  stream compaction.
* Compare scenes which are open (like the given cornell box) and closed
  (i.e. no light can escape the scene). Again, compare the performance effects
  of stream compaction! Remember, stream compaction only affects rays which
  terminate, so what might you expect?

  *Because rays terminate when they hit a light, or do not intersect with anything, in a closed room, there will not be any rays that do not intersect anything.  Thus, less rays will be terminated.  It is because of this that the stream compaction will not be used as much as compared to an open room.  We would guess that stream compaction does not help speed up the running time in a closed room as much as it does in an open one.  Test results helped to prove this answer.  


