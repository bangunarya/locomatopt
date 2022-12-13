import numpy as np
import os
from locomatopt.sampling import UniformRandom 
from locomatopt.metric import params_matrix
from locomatopt.gradientdescent import GradDescent
from locomatopt.auglmethod import ALM
from locomatopt.matrix import MatrixSH, MatrixWigner, MatrixSNF
from locomatopt.metric import coherence
from typing import Tuple, Optional, List


def coherence_optimization(
    basis: str,
    m: np.ndarray,
    params_mat: dict,
    params_grad: dict,
    step_size: Optional[float] = None,
     ) -> Tuple[List[float]]:
    """
    Parameters
    ----------
    m
        Array number of samples
    params_mat
        Dictionary parameters to construct a matrix
    params_grad
        Dictionary paramters for the gradient
    step_size
        Step size for gradient descent or ALM

    Returns
    -------
    object, list_of_errors
       
    """
    gd_ang_sample = []
    alm_ang_sample = []

    # Allocation coherence for samples
    alm_coh_sample = []
    gd_coh_sample = []

    # Loop for samples
    for idx in range(len(m)):
        
        # Generate samples
       
        unif_rand = UniformRandom(m=m[idx], basis=basis)

        # Gradient 
        gradesc = GradDescent(params_mat=params_mat, params_grad=params_grad)
        res_gradesc = gradesc.run_algo(angles=unif_rand.generate_angles(), step_size=step_size)
        
        select_mat = {'sh': MatrixSH,
                      'wigner': MatrixWigner,
                      'snf': MatrixSNF}
        
        mat_gd = select_mat[params_mat['types']](B=params_mat['B'], angles=res_gradesc['angle'],
                                                 case=params_mat['case'])
                       
        gd_ang_sample.append(res_gradesc['angle'])
        gd_coh_sample.append(res_gradesc['coherence'])
        
        # ALM 
        alm = ALM(params_mat=params_mat, params_grad=params_grad)
        res_alm = alm.run_algo(angles=unif_rand.generate_angles(), step_size=step_size)
        
        mat_alm = select_mat[params_mat['types']](B=params_mat['B'], angles=res_alm['angle'],
                                                  case=params_mat['case'])
        
        print('Sample (m): ', m[idx], ' GD ', res_gradesc['coherence'], coherence(mat_gd.normA))
        print('Sample (m): ', m[idx], ' ALM;', res_alm['coherence'], coherence(mat_alm.normA))
            
        alm_ang_sample.append(res_alm['angle'])
        alm_coh_sample.append(res_alm['coherence'])
        
    total_result = {'m': m, 'B': params_mat['B'], 'N': params_mat['N'],
                    'alm_ang': alm_ang_sample, 'alm_coh': alm_coh_sample,
                    'gd_ang': gd_ang_sample, 'gd_coh': gd_coh_sample}

    return total_result


if __name__ == '__main__':
    # Bandwidth
    B = 10

    # Antenna or normal expansion
    case = 'normal'

    # Sphere or Rotation Group
    types = 'sh'
    # Generate dimension
    N, col_comb = params_matrix(types=types, B=B, case=case)
    print(N)
    # Generate sampling points
    m = np.arange(21, N, 4).astype(np.int64)  # Samples
    
    # Matrix Parameters
    params_mat = {'B': B, 'N': N, 'types': types,
                  'col_comb': col_comb, 'case': case}
    p = 9
    # Gradient Parameters
    params_grad = {'max_iter': 300, 'eps': 1e-6,
                   'update': 'update_all', 'p_norm': p}

    # Path
    folder = os.path.join('results/', case, types)
    path = os.path.join(os.getcwd(), folder)
    filename = types + '' + case + '' + params_grad['update'] + '_alm'
    step_size = None
    total_result = coherence_optimization(basis=types, m=m, params_mat=params_mat,
                                          params_grad=params_grad, step_size=step_size)

    # Store Files
    os.makedirs(path, exist_ok=True)
    path_file = os.path.join(path, filename + '.npy')
    np.save(path_file, total_result)
