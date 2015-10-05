#!/usr/bin/env python3

from sys import argv

script, filename, in_depth = argv

txt = open(filename)

maxdepth = int(in_depth)
intersect_time  = [0.0] * maxdepth
compact_time = [0.0] * maxdepth
pixels = [  0] * maxdepth
iters = 0

line = txt.readline().split()
while len(line) > 0:
    depth = int(line[0])
    if depth == 0:
        iters += 1
    intersect_time[depth] += float(line[1])
    compact_time[depth] += float(line[2])
    pixels[depth] += int(line[3])
    line = txt.readline().split()

print("depth, intersect time, compact time, live rays")
for d, ti, tc, p in zip(range(maxdepth), intersect_time, compact_time, pixels):
    avg_i_time = ti / float(maxdepth)
    avg_c_time = tc / float(maxdepth)
    avg_live   = p / float(maxdepth)
    print("%d, %f, %f, %f" % (d, avg_i_time, avg_c_time, avg_live))
