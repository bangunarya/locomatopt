from .matrix import MatrixSNF
from .matrix_jax import wigner_snf_matrix, params_wigner_D_func, recurence_index
from jax import grad, jit 
import optax
import numpy as np
import jax.numpy as jnp 
from typing import Tuple, Optional, Union, List
 

def compute_loss(ang:Tuple, deg_order:Tuple, params:Tuple):
    """
    Function to compute loss of coherence matrix

    Parameters
    ----------
    ang
        Angles that we want to optimize
    deg_order
        Combination of degree and orders
    params
        Parameters that is needed to construct the matrix

    Returns
    -------
        Estimated coherence of the matrix
    """
    normA = wigner_snf_matrix(ang, deg_order, params)
   
    return loss_coherence(normA)


@jit
def loss_coherence(mat:jnp.ndarray):
    
    """
    Function to compute loss of coherence matrix

    Parameters
    ----------
    mat
        Matrix that we calculate the coherence
  

    Returns
    -------
        Estimated coherence of the matrix
    """
    
    # Calculate gram matrix
    gram = jnp.abs(jnp.dot(mat.conj().T, mat))
        
    return jnp.max(jnp.triu(gram,1))

def step(ang:Tuple, optimizer:optax.GradientTransformation, deg_order:Tuple, params:Tuple):

    """
    Function that is used to define step for gradient update

    Parameters
    ----------
    ang
        Angles that we want to optimize
    
    optimizer
        Defined optimization solver from optax
    
    deg_order
        Combination of degree and orders
    
    params
        Parameters that is used to construct SNF matrix

    Returns
    -------
    Updated angles

    """
    
    opt_state = optimizer.init(ang)
    
    gradient = grad(compute_loss, argnums=0)(ang, deg_order, params)
    
    updates, opt_state = optimizer.update(gradient, opt_state)
    
    
    return optax.apply_updates(ang, updates)



def run_algo(ang: Tuple, num_iter: int, optimizer:optax.GradientTransformation, deg_order:Tuple):

    """
    Function that is used run gradient descent based for coherence optimization

    Parameters
    ----------
    ang
        Angles that we want to optimize
    
    num_iter
        Number of iteration we want to run the algorithm
    
    step_size
        The step size for gradient approach
    
    deg_order
        Combination of degree and order polynomials


    Returns
    -------
    Optimized angles

    """
    
    #opt_init, opt_update, get_param = optimizers.nesterov(step_size=mu, mass=0.9) 
    #optimizer = optax.sgd(step_size) 

    # Initial coherence for reference
    coh_ref = 1
    # Initial angles for reference
    res = ang
    
    # Parameters to construct matrix
    weight_plus, deg_order_plus = params_wigner_D_func(deg_order, 1)
    idx_rec_plus, idx_deg_gr_one_plus = recurence_index(deg_order_plus)
    
    weight_min, deg_order_min = params_wigner_D_func(deg_order, -1)
    idx_rec_min, idx_deg_gr_one_min = recurence_index(deg_order_min)
    
    params = (weight_plus, weight_min, deg_order_plus, deg_order_min,
              idx_rec_plus, idx_deg_gr_one_plus, idx_rec_min, idx_deg_gr_one_min)
    
    # Rin iteration
    for it in range(num_iter):

        # Generate step update
        ang = step(ang, optimizer, deg_order, params) 
        
        # Construct a matrix for SNF
        mat = wigner_snf_matrix(ang, deg_order, params)
        
        # Calculate Optimized coherence
        coh = loss_coherence(mat)
        
        # Update the coherence if it is better 
        print('Iter ', it, 'Coherence', coh)
        if coh < coh_ref:

            coh_ref = coh 
            res = ang 
            
            #angles = {'theta':res[0],'phi':res[1], 'chi':(jnp.arange(len(ang[0])) % 2)*(np.pi/2)}
            #mat2 = MatrixSNF(jnp.max(deg_order[0]), angles, 'snf')
            #print(jnp.max(jnp.abs(mat2.normA.conj().T@mat2.normA - jnp.eye(mat2.normA.shape[1]))))
            print('Iteration ', it, 'Coherence', coh_ref)
 
    return res, coh_ref