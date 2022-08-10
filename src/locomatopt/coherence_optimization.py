from locomatopt.sampling import UniformRandom 
from locomatopt.metric import params_matrix
from locomatopt.gradientdescent import GradDescent
from locomatopt.auglmethod import ALM
from locomatopt.matrix import MatrixSH, MatrixWigner, MatrixSNF
from locomatopt.metric import coherence
from typing import Optional, Union
from typing import TypedDict
from typing_extensions import Literal
Mode = Union[Literal['normal'], Literal['antenna']] 
Basis = Union[Literal['sh'], Literal['wigner'], Literal['snf']]
Update = Union[Literal['update_all'], Literal['fix_theta']]


class GradParams(TypedDict):
    max_iter: int
    eps: float
    update: str
    step_size: Optional[float]
    p_norm: int


def get_params_grad(max_iter: int, eps: float, update: Update, 
                    p_norm: int, step_size: Optional[float] = None) -> GradParams:

    return {"max_iter": max_iter, 
            "eps": eps,
            "update": update,
            "p_norm": p_norm,
            "step_size": step_size
            }


class MatrixParams(TypedDict):
    B: int
    mode: str
    types: str
    N: int
    col_comb: list


def get_params_matrix(bandwidth: int, mode: Mode, basis: Basis) -> MatrixParams:

    """ 
    Return parameter matrix
    """
    column_dimension, col_comb = params_matrix(basis, bandwidth, mode)
    return {"B": bandwidth, 
            "N": column_dimension,
            "types": basis,
            "col_comb": col_comb,
            "case": mode
            }


def coherence_optimization_grad_descent(
    bandwidth: int,
    mode: Mode,
    basis: Basis,
    max_iter: int,
    eps: float,
    update: Update,
    p_norm: int,
    samples: list,
     ) -> dict:

    """
    Coherence minimization with gradient descent method

    Parameters
    ----------
    total_samples
        Generate several number of samples
    bandwidth
        Banwidth of our expansion to determine dimension of column matrix
    mode
        Evaluate normal expansion or expansion in terms of spherical near-field antenna measurements
    basis
        Choose basis function to construct matrix, spherical harmonics, Wigner D-functions, 
        or spherical near-field
    max_iter
        Maximum iteration for algorithms
    eps
        The error tolerance for the algorithms
    update  
        Evaluate update for some variables on the sphere (azimuth) or all 
        (elevation, azimuth and polarization)
    p_norm  
        Large enough p-norm to approximate infinity norm
   
    Returns
    -------
    object, list_of_errors
       
    """

    # Generate parameters
    params_mat = get_params_matrix(bandwidth=bandwidth, 
                                   mode=mode,
                                   basis=basis)

    params_grad = get_params_grad(max_iter=max_iter, 
                                  eps=eps, 
                                  update=update, 
                                  p_norm=p_norm)

    # Allocation coherence for samples
    gd_ang_sample = []
    gd_coh_sample = []

    # Loop for samples
    for idx in range(len(samples)):
        
        # Generate samples
       
        unif_rand = UniformRandom(m=samples[idx], basis=basis)

        # Gradient descent method
        gradesc = GradDescent(params_mat=params_mat, params_grad=params_grad)
        res_gradesc = gradesc.run_algo(angles=unif_rand.generate_angles(), 
                                       step_size=params_grad['step_size'])
        
        select_mat = {'sh': MatrixSH,
                      'wigner': MatrixWigner,
                      'snf': MatrixSNF}
        
        mat_gd = select_mat[params_mat['types']](B=params_mat['B'], angles=res_gradesc['angle'],
                                                 case=params_mat['case'])
                       
        gd_ang_sample.append(res_gradesc['angle'])
        gd_coh_sample.append(res_gradesc['coherence'])
                 
        print('Sample (m): ', samples[idx], ' GD ', res_gradesc['coherence'],
              coherence(mat_gd.normA))
        
    total_result = {'m': samples, 'B': params_mat['B'], 'N': params_mat['N'],
                    'gd_ang': gd_ang_sample, 'gd_coh': gd_coh_sample}

    return total_result
 

def coherence_optimization_alm(
    bandwidth: int,
    mode: Mode,
    basis: Basis,
    max_iter: int,
    eps: float,
    update: Update,
    p_norm: int,
    samples: int,
     ) -> dict:
    """
    Coherence minimization with augmented Lagrangian method
    Parameters
    ----------
    total_samples
        Generate several number of samples
    bandwidth
        Banwidth of our expansion to determine dimension of column matrix
    mode
        Evaluate normal expansion or expansion in terms of spherical near-field antenna measurements
    basis
        Choose basis function to construct matrix, spherical harmonics, Wigner D-functions, 
        or spherical near-field
    max_iter
        Maximum iteration for algorithms
    eps
        The error tolerance for the algorithms
    update  
        Evaluate update for some variables on the sphere (azimuth) or all 
        (elevation, azimuth and polarization)
    p_norm  
        Large enough p-norm to approximate infinity norm
   
    Returns
    -------
    object, list_of_errors
       
    """

    # Generate parameters
    params_mat = get_params_matrix(bandwidth=bandwidth, 
                                   mode=mode,
                                   basis=basis)

    params_grad = get_params_grad(max_iter=max_iter, 
                                  eps=eps, 
                                  update=update, 
                                  p_norm=p_norm)

    # Allocation coherence for samples
    alm_ang_sample = []
    alm_coh_sample = []
 
    # Loop for samples
    for idx in range(len(samples)):
        
        # Generate samples
       
        unif_rand = UniformRandom(m=samples[idx], basis=basis)
 
        select_mat = {'sh': MatrixSH,
                      'wigner': MatrixWigner,
                      'snf': MatrixSNF}
        
        # Augmented Lagrangian method
        alm = ALM(params_mat=params_mat, params_grad=params_grad)
        res_alm = alm.run_algo(angles=unif_rand.generate_angles(), 
                               step_size=params_grad['step_size'])
        
        mat_alm = select_mat[params_mat['types']](B=params_mat['B'], angles=res_alm['angle'],
                                                  case=params_mat['case'])
         
        print('Sample (m): ', samples[idx], ' ALM;', res_alm['coherence'], coherence(mat_alm.normA))
            
        alm_ang_sample.append(res_alm['angle'])
        alm_coh_sample.append(res_alm['coherence'])
        
    total_result = {'m': samples, 'B': params_mat['B'], 'N': params_mat['N'],
                    'alm_ang': alm_ang_sample, 'alm_coh': alm_coh_sample}

    return total_result
