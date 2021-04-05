# Nanophysics2

## Requirements
* OpenMP
* armadillo

## Build
Either use Makefile, or manually:
```
g++ condexp.cpp -o condexp --std=c++11 -O3 -larmadillo -fopenmp
```

## Usage
```
./condexp [alpha] [num]
```

alpha - alpha to compute conductances for
num - number of random impurity coherent calculations to run
    - if num<=0, calculates incoherent result instead.
