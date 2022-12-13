from .matrix import MatrixSNF
from .prox import project_l1_ball
from .matrix_jax import wigner_snf_matrix, params_wigner_D_func, recurence_index
from jax import grad, jit 
import optax
import numpy as np
import jax.numpy as jnp 
from typing import Tuple, Optional, Union, List


@jit
def loss(z_aux:jnp.ndarray, vect_coh:jnp.ndarray, u_dual:jnp.ndarray, rho:float):
    """
    Loss function for augmented Lagrangian method

    Parameters
    ----------
   
             
    vect_coh
        Vectorization of the coherence
    
    z_aux,u_dual
        Augmented Lagrangian variables, see the paper
    
    rho
        Augmented Lagrangian variables, see the paper


    Return
    ------
    Calculated loss functions
    """
    
    return (rho/2)*jnp.linalg.norm(z_aux - (vect_coh - u_dual),2)**2 

def compute_loss(ang:Tuple, deg_order:Tuple, params:Tuple, 
                 z_aux:jnp.ndarray, u_dual:jnp.ndarray, rho:float):
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

    z_aux,u_dual
        Augmented Lagrangian variables, see the paper
    
    rho
        Augmented Lagrangian variables, see the paper


    Returns
    -------
        Estimated coherence of the matrix
    """
    normA = wigner_snf_matrix(ang,deg_order, params)
    
    vect_coh = jnp.triu(jnp.abs(jnp.dot(normA.conj().T, normA)),1).flatten("C")
    
    return loss(z_aux, vect_coh, u_dual, rho)



 
def update_prox(optimizer:optax.GradientTransformation, vect_coh:jnp.ndarray, u_dual:jnp.ndarray, 
                z_aux:jnp.ndarray, rho:float, step_size:float):
    """
    Perform single step for proximal method, infinity norm is the projection
    to the l1 ball, see paper for explanation

    Arguments:


        - z,w : array
            Augmented Lagrangian variables, see the paper

        - rho : float
            Augmented Lagrangian variables, see the paper

    Returns:
        - vX : dict
            projected vector after proximal of infinity norm
    """
    # Initialize input
    opt_state = optimizer.init(z_aux)
    
    # Calculate gradient
    gradient =  grad(loss, argnums=0)(z_aux, vect_coh, u_dual, rho) 
    
    # Update the result
    updates, opt_state = optimizer.update(gradient, opt_state)
    vV = optax.apply_updates(z_aux, updates)
    
    # Proximal method to project into l1
 
    paramLambda = 10
    stopThr = 1e-4
    vX = vV - (step_size*paramLambda*project_l1_ball(vV/(paramLambda*step_size), 1, 
               stopThr))

    return jnp.array(vX)


 
def update_ang(optimizer:optax.GradientTransformation,
               u_dual:jnp.ndarray, z_aux:jnp.ndarray, rho:float, 
               angles:Tuple, deg_order:Tuple, params:Tuple):
    
     
    # Initialize input
    opt_state = optimizer.init(angles)

    # Calculate gradient
    gradient =  grad(compute_loss, argnums=0)(angles, deg_order, params, z_aux, u_dual, rho) 
    
    # Update the result
    updates, opt_state = optimizer.update(gradient, opt_state)
    
     
    return optax.apply_updates(angles, updates) 


def params_alm(angles:Tuple, deg_order:Tuple):

    """
    Parameters for augmented Lagrangian algorithms
    Arguments:
        - angles : dict
            angle to construct matrix

    Returns:
        - mat : dict
            updated constructed matrix

        - coh : float
            coherence of current matrix
        - vect_coh ; array
            vectorization of the coherence
        - grad : dict
            gradient with respect to angles  (theta, phi, chi)
        - z,u : array
            Augmented Lagrangian variables, see the paper
        - rho: float
            Augmented Lagrangian variables, see the paper

    """
    
    weight_plus, deg_order_plus = params_wigner_D_func(deg_order, 1)
    idx_rec_plus, idx_deg_gr_one_plus = recurence_index(deg_order_plus)
    
    weight_min, deg_order_min = params_wigner_D_func(deg_order, -1)
    idx_rec_min, idx_deg_gr_one_min = recurence_index(deg_order_min)
    
    params = (weight_plus, weight_min, deg_order_plus, deg_order_min,
              idx_rec_plus, idx_deg_gr_one_plus, idx_rec_min, idx_deg_gr_one_min)
    
    # Vectorize the product
    mat = wigner_snf_matrix(angles, deg_order, params)

    vect_coh = jnp.triu(jnp.abs(jnp.dot(mat.conj().T, mat)),1).flatten("C")

    # Initial coherence
    coh = jnp.max(vect_coh)

    # Dual
    u_dual = jnp.array(np.random.rand(len(vect_coh))) 

    # Aux variables
    z_aux = jnp.array(np.random.rand(len(vect_coh)))  

    rho = 1

    return vect_coh, coh, u_dual, z_aux, rho, params

def run_algo(ang:Tuple, num_iter:int,optimizer:optax.GradientTransformation, 
             step_size:float, deg_order:Tuple):
    
    
       # Initial angle    
    res = ang

    # Construct initialization    
    vect_coh, coh_ref, u_dual, z_aux, rho, params = params_alm(ang, deg_order)
    
    for it in range(num_iter):
        
        # Update z from proximal
        z_aux = update_prox(optimizer, vect_coh, u_dual, z_aux, rho, step_size)
        
        # Update angles
        ang = update_ang(optimizer, u_dual, z_aux, 
                         rho, ang, deg_order, params)

        # Vectorize the product
        mat = wigner_snf_matrix(ang,deg_order, params)
    
        vect_coh = jnp.triu(jnp.abs(jnp.dot(mat.conj().T, mat)),1).flatten("C")
        
        # Update u
        u_dual = u_dual + rho*(z_aux - vect_coh)
        
        coh = jnp.max(vect_coh)
        
        # Store 
      
        if coh < coh_ref:
            coh_ref = coh 
          
            res = ang

            #angles = {'theta':res[0],'phi':res[1], 'chi':(jnp.arange(len(ang[0])) % 2)*(np.pi/2)}
            #mat2 = MatrixSNF( jnp.max(deg_order[0]), angles, 'snf')
            #print(jnp.max(jnp.abs(mat2.normA.conj().T@mat2.normA - jnp.eye(mat2.normA.shape[1]))))
            print('Iteration ', it, 'Coherence', coh_ref)
        
    return res, coh_ref