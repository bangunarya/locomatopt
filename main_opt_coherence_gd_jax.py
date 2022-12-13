from locomatopt.matrix_jax import degree_orders_snf
from locomatopt.gradientdescent_jax import run_algo
from locomatopt.sampling import UniformRandom, PoleRandom
import optax
from jax import grad, jit 
import jax.numpy as jnp
import numpy as np 
from typing import Tuple, Optional, Union, List
import os
from jax.config import config; 
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
def coherence_optimization(
    basis: str,
    m: np.ndarray,
    num_iter: int,
    optimizer: optax.GradientTransformation,
    step_size: float,
    deg_order: Tuple,
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
    gd_ang_sample_jax = []

    # Allocation coherence for samples
    gd_coh_sample_jax = []

    # Loop for samples
    for idx in range(len(m)):
        
        # Generate samples
        unif_rand = UniformRandom(m=m[idx], basis=basis).generate_angles()
        ang = (unif_rand['theta'], unif_rand['phi'])

        # Run algorithm
        res_ang, coh = run_algo(ang, num_iter, optimizer, deg_order)
       
                       
        gd_ang_sample_jax.append(res_ang)
        gd_coh_sample_jax.append(coh)
        
        
        
        print('Sample (m): ', m[idx], ' GD ', 'Coherence', coh)
     
        
    total_result = {'m': m, 'B': jnp.max(deg_order[0]),
                    'gd_ang': gd_ang_sample_jax, 'gd_coh': gd_coh_sample_jax}

    return total_result

if __name__ == '__main__':

    case = 'antenna'
    basis = 'snf'
    B = 73
    ll, k = degree_orders_snf(B)
    deg_order = (ll,k)
    

    m = np.arange(1000, 10000,1375)
    

    step_size =  5e-1
    optimizer = optax.sgd(step_size) 
    num_iter = 100

    # Path
    folder = os.path.join('results/', case, basis)
    path = os.path.join(os.getcwd(), folder)
    filename = basis + '' + case + '' + 'update_all' + '_gd_jax'
    total_result = coherence_optimization(basis=basis, m=m, num_iter=num_iter,
                                          optimizer=optimizer, step_size=step_size,
                                          deg_order=deg_order)

    # Store Files
    os.makedirs(path, exist_ok=True)
    path_file = os.path.join(path, filename + '.npy')
    np.save(path_file, total_result)
