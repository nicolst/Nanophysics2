# Nanophysics2
Calculates the (in)coherent conductance of a train of scatterers and free propagation areas, for N channels (set in file). Homework in TFY4340 Nanophysics at NTNU.
![bitmap](https://user-images.githubusercontent.com/35232838/116367326-87b5ad00-a807-11eb-93c8-4f7511b63b7f.png)

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

- alpha - alpha to compute conductances for
- num
    - number of random impurity coherent calculations to run
    - if num<=0, calculates incoherent result instead.
