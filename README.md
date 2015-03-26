# SusliX Lite

## Overview

This is a 2D physics engine with SoA/SIMD optimizations.

The optimization effort is focused on contact solver - all other parts did not receive much attention.
For example, it uses a bruteforce broadphase to keep code simple.

The engine is written by Alexander Sannikov aka Suslik &lt;i.make.physics at gmail.com&gt;

The optimizations are implemented by Arseny Kapoulkine &lt;arseny.kapoulkine@gmail.com&gt;

The code is licensed under the MIT license.

## Performance

Run 'make profile' to benchmark the solver in various modes.
