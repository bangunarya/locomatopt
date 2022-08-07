# Low Coherence Matrix Optimization

This repository contains the code for the optimization of sampling points on the sphere and the rotation group that yield low-coherence matrices constructed from spherical harmonics and Wigner D-functions. The results and algorithms have been published in the following papers:


[Sensing Matrix Design and Sparse Recovery on the Sphere and the Rotation Group](https://arxiv.org/abs/1904.11596)\
[Tight bounds on the mutual coherence of sensing matrices for Wigner D-functions on regular grids](https://link.springer.com/article/10.1007/s43670-021-00006-2)\
[Optimizing Sensing Matrices for Spherical Near-Field Antenna Measurements](https://arxiv.org/abs/2206.02181)\

#Requirements

The following Python libraries are required to run the code in this repository:

```
numpy
scipy
```
and can be installed with pip install -e .

# Usage
All the figures in the paper can be reproduced by running the respective notebooks as indicated below:

## Citation
```
@article{bangun2020sensing,
  title={Sensing matrix design and sparse recovery on the sphere and the rotation group},
  author={Bangun, Arya and Behboodi, Arash and Mathar, Rudolf},
  journal={IEEE Transactions on Signal Processing},
  volume={68},
  pages={1439--1454},
  year={2020},
  publisher={IEEE}
}

@article{bangun2021tight,
  title={Tight bounds on the mutual coherence of sensing matrices for Wigner D-functions on regular grids},
  author={Bangun, Arya and Behboodi, Arash and Mathar, Rudolf},
  journal={Sampling Theory, Signal Processing, and Data Analysis},
  volume={19},
  number={2},
  pages={1--39},
  year={2021},
  publisher={Springer}
}

@article{bangun2022optimizing,
  title={Optimizing Sensing Matrices for Spherical Near-Field Antenna Measurements},
  author={Bangun, Arya and Culotta-L{\'o}pez, Cosme},
  journal={arXiv preprint arXiv:2206.02181},
  year={2022}
}
```
## Licence
All files are provided under the terms of the MIT License
