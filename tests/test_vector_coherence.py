import numpy as np
import pytest
from locomatopt.sampling import SpiralSampling, UniformRandom, PoleRandom
from locomatopt.matrix import MatrixSH, MatrixWigner, MatrixSNF
from locomatopt.vectorizing_coh import vector_coherence_sh, vector_coherence_wigner, vector_coherence_snf, matrix_coherence
from locomatopt.metric import params_matrix

 
def test_vect_sh():
    
    m = 100  
    B = 10
    types = 'sh'
    case = 'normal'
    N, col_comb = params_matrix(types=types, B=B, case=case)
    unif_rand = UniformRandom(m=m, basis=types)
    angles = unif_rand.generate_angles()
    mat_sh = MatrixSH(B=B, angles=angles,
                      case=case)
    
    # Matrix Parameters
    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}
    vec_coh_sh = vector_coherence_sh(params_mat, mat_sh)
    mat_coh_sh = matrix_coherence(mat_sh)
   
    assert np.allclose(len(vec_coh_sh),len(mat_coh_sh))
    assert np.allclose(vec_coh_sh,  mat_coh_sh)       


def test_vect_wigner():
    
    m = 81  
    B = 4
    types = 'wigner'
    case = 'normal'
    N, col_comb = params_matrix(types=types, B=B, case=case)
    unif_rand = UniformRandom(m=m, basis=types)
    angles = unif_rand.generate_angles()
    mat_wigner = MatrixWigner(B=B, angles=angles,
                              case=case)
    
    # Matrix Parameters
    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}
    vec_coh_wigner = vector_coherence_wigner(params_mat, mat_wigner)
    mat_coh_wigner = matrix_coherence(mat_wigner)
   
    assert np.allclose(len(vec_coh_wigner),len(mat_coh_wigner))
    assert np.allclose(vec_coh_wigner,  mat_coh_wigner)       


def test_vect_snf():
    
    m = 100  
    B = 8
    types = 'snf'
    case = 'normal'
    N, col_comb = params_matrix(types=types, B=B, case=case)
    unif_rand = UniformRandom(m=m, basis=types)
    angles = unif_rand.generate_angles()
    mat_snf = MatrixSNF(B=B, angles=angles,
                        case=case)
    
    # Matrix Parameters
    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}
    vec_coh_snf = vector_coherence_snf(params_mat, mat_snf)
    mat_coh_snf = matrix_coherence(mat_snf)
   
    assert np.allclose(len(vec_coh_snf), len(mat_coh_snf))
    assert np.allclose(vec_coh_snf,  mat_coh_snf)


@pytest.mark.benchmark()
def test_bench_vectcoh_sh(benchmark):
    m = 100  
    B = 10
    types = 'sh'
    case = 'normal'
    N, col_comb = params_matrix(types=types, B=B, case=case)
    unif_rand = UniformRandom(m=m, basis=types)
    angles = unif_rand.generate_angles()
    mat_sh = MatrixSH(B=B, angles=angles,
                      case=case)
    
    # Matrix Parameters
    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}
    benchmark(lambda: vector_coherence_sh(params_mat, mat_sh))


@pytest.mark.benchmark()
def test_bench_matcoh_sh(benchmark):
    m = 100  
    B = 10
    types = 'sh'
    case = 'normal'
    unif_rand = UniformRandom(m=m, basis=types)
    angles = unif_rand.generate_angles()
    mat_sh = MatrixSH(B=B, angles=angles,
                      case=case)
    benchmark(lambda: matrix_coherence(mat_sh))


@pytest.mark.benchmark()
def test_bench_vectcoh_wigner(benchmark):
    m = 75  
    B = 4
    types = 'wigner'
    case = 'normal'
    N, col_comb = params_matrix(types=types, B=B, case=case)
    unif_rand = UniformRandom(m=m, basis=types)
    angles = unif_rand.generate_angles()
    mat_wigner = MatrixWigner(B=B, angles=angles,
                              case=case)
    
    # Matrix Parameters
    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}
    benchmark(lambda: vector_coherence_wigner(params_mat, mat_wigner))


@pytest.mark.benchmark()
def test_bench_matcoh_wigner(benchmark):
    m = 75
    B = 4
    types = 'wigner'
    case = 'normal'
    unif_rand = UniformRandom(m=m, basis=types)
    angles = unif_rand.generate_angles()
    mat_wigner = MatrixWigner(B=B, angles=angles,
                              case=case)
    benchmark(lambda: matrix_coherence(mat_wigner))


@pytest.mark.benchmark()
def test_bench_vectcoh_snf(benchmark):
    m = 100  
    B = 8
    types = 'snf'
    case = 'normal'
    N, col_comb = params_matrix(types=types, B=B, case=case)
    unif_rand = UniformRandom(m=m, basis=types)
    angles = unif_rand.generate_angles()
    mat_snf = MatrixSNF(B=B, angles=angles,
                        case=case)
    
    # Matrix Parameters
    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}
    benchmark(lambda: vector_coherence_snf(params_mat, mat_snf))

@pytest.mark.benchmark()
def test_bench_matcoh_snf(benchmark):
    m = 100  
    B = 8
    types = 'snf'
    case = 'normal'
    unif_rand = UniformRandom(m=m, basis=types)
    angles = unif_rand.generate_angles()
    mat_snf = MatrixSNF(B=B, angles=angles,
                        case=case)
    benchmark(lambda: matrix_coherence(mat_snf))
