from locomatoptpy.sampling import UniformRandom
from locomatoptpy.gradient import GradSH, GradWigner, GradWignerSNF
from locomatoptpy.gradientcopy import GradSH as GSH
from locomatoptpy.gradientcopy import GradWigner as GW
from locomatoptpy.gradientcopy import GradWignerSNF as GWS
from locomatoptpy.matrix import MatrixSH, MatrixWigner, MatrixSNF
from locomatoptpy.metric import params_matrix
import numpy as np


def test_gradient_sh():

    # Bandwidth
    B = 10
    # Antenna or normal expansion
    case = 'normal'
    # Sphere or Rotation Group
    types = 'sh'  
    N, col_comb = params_matrix(types=types, B=B, case=case)

    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}

    # Generate sampling points
    m = 70 
    unifrand = UniformRandom(m=m, basis=types)  # Wigner polar should random
    angles = unifrand.generate_angles()

    mat_sh = MatrixSH(B=B, angles=angles,
                      case=case)

    p = 9
    # Gradient Parameters
    params_grad = {'max_iter':  50, 'eps': 1e-6,
                   'update': 'fix_theta', 'p_norm': p}

    grad1 = GradSH(matrix=mat_sh, col_comb=params_mat['col_comb'],
                   p=params_grad['p_norm'])   

    grad2 = GSH(matrix=mat_sh, col_comb=params_mat['col_comb'],
                p=params_grad['p_norm'])   
    grad_total = grad2.grad_total()
 
    assert np.allclose(grad1.gr_theta, grad_total['theta'])
    assert np.allclose(grad1.gr_theta_total, grad_total['theta_func'])
    assert np.allclose(grad1.gr_phi, grad_total['phi'])
    assert np.allclose(grad1.gr_phi_total, grad_total['phi_func'])


def test_gradient_wigner():

    # Bandwidth
    B = 4
    # Antenna or normal expansion
    case = 'normal'
    # Sphere or Rotation Group
    types = 'wigner'  
    N, col_comb = params_matrix(types=types, B=B, case=case)

    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}

    # Generate sampling points
    m = 70 
    unifrand = UniformRandom(m=m, basis=types)  # Wigner polar should random
    angles = unifrand.generate_angles()

    mat_wigner = MatrixWigner(B=B, angles=angles,
                              case=case)

    p = 9
    # Gradient Parameters
    params_grad = {'max_iter':  50, 'eps': 1e-6,
                   'update': 'fix_theta', 'p_norm': p}

    grad1 = GradWigner(matrix=mat_wigner, col_comb=params_mat['col_comb'],
                       p=params_grad['p_norm'])   

    grad2 = GW(matrix=mat_wigner, col_comb=params_mat['col_comb'],
               p=params_grad['p_norm'])   
    grad_total = grad2.grad_total()
 
    assert np.allclose(grad1.gr_theta, grad_total['theta'])
    assert np.allclose(grad1.gr_theta_total, grad_total['theta_func'])
    assert np.allclose(grad1.gr_phi, grad_total['phi'])
    assert np.allclose(grad1.gr_phi_total, grad_total['phi_func'])
    assert np.allclose(grad1.gr_chi, grad_total['chi'])
    assert np.allclose(grad1.gr_chi_total, grad_total['chi_func'])


def test_gradient_snf():

    # Bandwidth
    B = 10
    # Antenna or normal expansion
    case = 'normal'
    # Sphere or Rotation Group
    types = 'snf'  
    N, col_comb = params_matrix(types=types, B=B, case=case)

    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}

    # Generate sampling points
    m = 70 
    unifrand = UniformRandom(m=m, basis=types)  # Wigner polar should random
    angles = unifrand.generate_angles()

    mat_snf = MatrixSNF(B=B, angles=angles,
                        case=case)

    p = 9
    # Gradient Parameters
    params_grad = {'max_iter':  50, 'eps': 1e-6,
                   'update': 'fix_theta', 'p_norm': p}

    grad1 = GradWignerSNF(matrix=mat_snf, col_comb=params_mat['col_comb'],
                          p=params_grad['p_norm'])   

    grad2 = GWS(matrix=mat_snf, col_comb=params_mat['col_comb'],
                p=params_grad['p_norm'])   
                
    grad_total = grad2.grad_total()
 
    assert np.allclose(grad1.gr_theta, grad_total['theta'])
    assert np.allclose(grad1.gr_theta_total, grad_total['theta_func'])
    assert np.allclose(grad1.gr_phi, grad_total['phi'])
    assert np.allclose(grad1.gr_phi_total, grad_total['phi_func'])
    assert np.allclose(grad1.gr_chi, grad_total['chi'])
    assert np.allclose(grad1.gr_chi_total, grad_total['chi_func'])