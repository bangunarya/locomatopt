from locomatopt.sampling import SpiralSampling
from locomatopt.matrix import MatrixSH, MatrixWigner, MatrixSNF
import numpy as np
import pytest
import os
import scipy.io as sio


def test_sh():
    # Bandwidth
    B = 10
    # Antenna or normal expansion
    case = 'normal'
    # Sphere or Rotation Group
    types = 'sh'  
    # Generate sampling points
    m = 70 
    spiral = SpiralSampling(m=m, basis=types)  # Wigner polar should random
    angles = spiral.generate_angles()
    mat_sh = MatrixSH(B=B, angles=angles,
                      case=case)

    path = os.path.join(os.getcwd(), 'test_matrix.mat')
    mat_contents = sio.loadmat(path)
    mat_SH_matlab = mat_contents['SH_spiral_70']
    
    assert np.allclose(mat_sh.A, mat_SH_matlab)


def test_wigner():
    # Bandwidth
    B = 4

    # Antenna or normal expansion
    case = 'normal'

    # Sphere or Rotation Group
    types = 'wigner' 
    
    # Generate sampling points
    m = 70

    spiral = SpiralSampling(m=m, basis=types) #Wigner polar should random
    angles = spiral.generate_angles()
    path = os.path.join(os.getcwd(), 'test_matrix.mat')
    mat_contents = sio.loadmat(path)
    mat_Wigner_matlab = mat_contents['Wigner_spiral']
    mat_wigner = MatrixWigner(B=B, angles=angles,
                              case=case)
    np.allclose(mat_wigner.A, mat_Wigner_matlab)


def test_wigner_snf():
    # Bandwidth
    B = 10

    # Antenna or normal expansion
    case = 'normal'

    # Sphere or Rotation Group
    types = 'snf'
   
    # Generate sampling points
    m = 70

    spiral = SpiralSampling(m=m, basis=types) # Wigner polar should random
    angles = spiral.generate_angles()

    mat_snf = MatrixSNF(B=B, angles=angles,
                        case=case)
    path = os.path.join(os.getcwd(), 'test_matrix.mat')
    mat_contents = sio.loadmat(path)
    mat_WignerSNF_matlab = mat_contents['Wigner_SNF']

    assert np.allclose(mat_snf.A, mat_WignerSNF_matlab)
  
        
