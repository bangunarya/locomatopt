from jax import lax, jit, vmap
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import gammaln
from typing import Tuple, Optional, Union, List
from functools import partial


def jacobi_false(tup_int: Tuple):

    """
    The condition function from lax if the index does not satisfy the condition.
    This function acts as identity function

    Parameters
    ----------
    tup_int
        Tuple that contains all parameters to generate polynomials

    Returns
    ------
    prior_recurrence
        Tuple contains the first two prior polynomials as well as alpha, beta (order Jacobi polynomials)
    prior_recurrence[1]
        The last updated polynomials
    """
    __, __, __,  __, __, prior_recurrence = tup_int
    return prior_recurrence, prior_recurrence[1]

def jacobi_true(tup_int: Tuple):
    
    """
    The condition function from lax if the index does not satisfy the condition.
    This function acts as function to calculate next recurrence

    Parameters
    ----------
    tup_int
        Tuple that contains all parameters to generate polynomials

    Returns
    ------
    prior_recurrence
        Tuple contains the first two prior polynomials as well as alpha, beta (order Jacobi polynomials)
    prior_recurrence[1]
        The last updated polynomials
    """
    const_2,const_3, x,  const_4, const_1, prior_recurrence = tup_int
    last_polynomials = ( ( const_3 + const_2  * x  )* prior_recurrence[1]  + const_4  * prior_recurrence[0] ) / const_1 
  
    next_recurrence = (prior_recurrence[1], last_polynomials , prior_recurrence[2], prior_recurrence[3])
    return next_recurrence, last_polynomials
 

def recurrence_jacobi(prior_recurrence: Tuple, 
                      idx_recurrence: jnp.ndarray, 
                      x: jnp.ndarray):
    """
    In order to construct the matrix for experimental SNF, Jacobi polynomials
    is required to construct Wigner D function, since Jax does not implement
    its scipy version, here is the implementation with recurrence relation to avoid
    numerical instability due to factorial from alternative implmentation.

    It should be noted that since we want to efficiently improve performance we implement
    the recurrence with lax.scan approach

    Parameters
    ----------
    prior_recurrence
        Tuple contains the first two prior polynomials as well as alpha, beta (order Jacobi polynomials)
    
    idx_recurrence
        Index array contains from range(1, n) where n is the maximum degree Jacobi polynomials

    x 
        Array of the interval betweeen [-1, 1], generated from cos(theta) 
    

    Returns
    -------
    next_recurrence
        Next two polynomials that is used for the next recurrence
    last polynom
        the last polynomials
    """
    
    # Get parameters
    __,__, alpha, beta = prior_recurrence
    
    # Constant defined to construct the polynomials
    const_1 = 2 * (idx_recurrence + 1) * ( idx_recurrence + alpha + beta + 1) * ( 2 * (idx_recurrence + 1) - 2 + alpha + beta )
    const_2 = ( 2 * (idx_recurrence + 1) - 1 + alpha + beta ) * ( 2 * (idx_recurrence + 1) + alpha + beta )* ( 2 * (idx_recurrence + 1) - 2 + alpha + beta )
    const_3 = ( 2 * (idx_recurrence + 1) - 1 + alpha + beta ) * ( alpha + beta ) * ( alpha - beta )
    const_4 = - 2 * ( (idx_recurrence + 1) - 1 + alpha ) * ( (idx_recurrence + 1) - 1 + beta )  * ( 2 * (idx_recurrence + 1) + alpha + beta )
 
    # Get condition if const_1 == 0, we pass the identity function, e.g., jacobi_false 
    next_recurrence, last_polynom = lax.cond(const_1, jacobi_true, jacobi_false, (const_2, const_3, x,  const_4, const_1,prior_recurrence))

    return next_recurrence, last_polynom 

def wigner_d_function_recurrence_lax(theta: jnp.ndarray, 
                                     idx: jnp.ndarray, 
                                     deg_order: Tuple):

    """
    Function to construct Wigner (small) d functions using lax.scan 
    to improve performance for constructing large matrix

    Parameters
    ----------
    theta
        Sampling points on the elevation
    
    idx
        Index arange to construct a matrix using lax.scan
    
    deg_order
        Degree and order polynomials as well as the index arange for recurrence 

    Returns
    -------
    theta
        Sampling points on the elevation (as carry in lax.scan)
    next_recurrence[1]
        Updated all jacobi polynomials in terms of list
    
    """
    
    # Get degree, orders and index arange of polynomials
    __, alpha, beta, idx_recurrence = deg_order

    # Construct Jacobi polynomials function
    part_recurrence_jacobi = partial(recurrence_jacobi, x = jnp.cos(theta)) 
    
    # Construct first two well known Jacobi polynomials for recurrence relation
    J2 = ( 1.0 + 0.5 * ( alpha[idx] + beta[idx] ) ) * jnp.cos(theta) + 0.5 * ( alpha[idx]- beta[idx] ) 
    J1 = jnp.ones_like(J2)

    prior_recurrence = (J1,J2, alpha[idx], beta[idx])
    
    id_recurrence = idx_recurrence[idx,:]
    
    #assert False
    next_recurrence, last_polynomials = lax.scan(part_recurrence_jacobi, prior_recurrence, id_recurrence)
 
    return theta, next_recurrence[1]

@jit
def fact_jnp(n:int):
    '''
    Construct factorial with jit

    Parameters
    ----------
    n 
        Integer value for factorial


    Returns
    -------
    
    Factorial result

    '''
    return jnp.exp(gammaln(n + 1))


def degree_orders_snf(B:int):

    '''
    Generating combination of degree and orders for Wigner D-function in
    spherical near-field measurements (SNF)

    Parameters
    ----------
    B 
        Bandwidth of the polynomials, maximum degree

    Returns
    -------
    ll
        Degree of polynomials
    k
        Order of polynomials
    '''
    N = 2*B**2 + 4*B
    ll=jnp.zeros(N//2, dtype=jnp.int64)
    k=jnp.zeros(N//2, dtype=jnp.int64)
    
    idx_beg = 0
    for l in jnp.arange(1,B + 1):
         
        idx_end = idx_beg + 2*l
        # Construct all orders -l,....,l
        k = k.at[idx_beg:idx_end+1].set(jnp.arange(-l,l+1))
        # Construct same degree for related order -l,..,l
        ll = ll.at[idx_beg:idx_end+1].set(jnp.full(2*l+1,l))
        # Update index
        idx_beg = idx_beg + 2*l+1
        
    
    return ll, k


def params_wigner_D_func(deg_order: Tuple, mu: int):
    
    '''
    Generating parameters that is required to construct
    the for Wigner D-function in spherical near-field measurements (SNF)

    Parameters
    ----------
    deg_order
        Combination of degree and order
    
    mu
        Fix order between +1 and  -1

    Returns
    -------
    weight_total
        Weight total that is used to generate Wigner D-functions
    
    
    '''
    ll = deg_order[0]
    k = deg_order[1]
   
    
  
    ## Parameters
    normalization = jnp.sqrt((2.0*ll+1)/(8.0*jnp.pi**2))


    eta = jnp.zeros_like(k) 
    eta = jnp.where(mu < k, eta, 1)
    eta = jnp.where(mu >= k, eta,np.float_power(-1,mu - k))
     
    ## Set Wigner d for positive
    mu_sign = jnp.abs(k - mu)
    vu_sign = jnp.abs(k + mu)
    s_sign = ll - (mu_sign + vu_sign)//2
    
  
    norm_gamma = jnp.sqrt((fact_jnp(s_sign)*fact_jnp(s_sign + mu_sign + vu_sign))/
                          (fact_jnp(s_sign + mu_sign)*(fact_jnp(s_sign + vu_sign))))

    
    weight_total = normalization*norm_gamma*eta
   
    #wigner_d = (weight_total*Plkn(s_sign,mu_sign,vu_sign,jnp.cos(theta)))
    return weight_total, (s_sign, mu_sign, vu_sign)


def matrix_params(theta:jnp.ndarray, deg_order:Tuple):

    '''
    Generating parameters that is used to construct matrix 
    from Wigner (small) d-functions

    Parameters
    ----------
    deg_order
        Combination of degree and order for Wigner d-functions (s, mu,vu)
        note it differs from (l,k)
    
    theta
        Sampling points on the elevation

    Returns
    -------
    matrix_allocation
        Matrix allocation for Wigner (small) d-functions
    
    
    '''
    
    # Alocate the matrix initial, polynomials with degree zero is one    
    matrix_allocation = jnp.ones((len(theta), len(deg_order[0])))
    idx_deg_one = jnp.where(deg_order[0] == 1)
    
    # Jacobi polynomial with degree 1
    J1 = ( 1.0 + 0.5 * (deg_order[1] + deg_order[2]))[jnp.newaxis,:] * jnp.cos(theta)[:,jnp.newaxis]  + 0.5 * ( deg_order[1] - deg_order[2] )[jnp.newaxis,:]
    
    # Construct well-known position
    matrix_allocation = matrix_allocation.at[:,idx_deg_one].set(J1[:, idx_deg_one])
    
   
   
    return matrix_allocation 

def recurence_index(deg_order: Tuple):

    """
    Function to generate recurrence index for jacobi polynomials.
    For each degree n, we construct arange(1,n) and set -1 for outside
    interval so that Jacobi polynomials wont construct polynomials

    Parameters
    ----------
    deg_order
        Combination of degree and order for Wigner d-functions (s, mu,vu)
        note it differs from (l,k)

    
    Returns
    -------
    idx_rec
        Construct index for recurrence relation

    idx_deg_gr_one
        Index where the computation of recurrence should be done, for degree > 1
        Since for degree 0 and 1 it is defined prior recurrence
    

    """
    # Get the index for degree greater than one, used for recurrence relation
    idx_deg_gr_one = jnp.where(deg_order[0] > 1)
   
    # Prepare the index for recurrence
    idx_rec = -jnp.ones((len(deg_order[0]),jnp.max(deg_order[0])-1))   
    
    for idx in idx_deg_gr_one[0]:
        idx_rec = idx_rec.at[idx,0:deg_order[0][idx]-1].set(jnp.arange(1,deg_order[0][idx]))
    return idx_rec, idx_deg_gr_one

def weight(theta:jnp.array, deg_order:Tuple):
    """
    Function to construct matrix for weight of Jacobi polyomials

    Parameters
    ----------
    deg_order
        Combination of degree and order for Wigner d-functions (s, mu,vu)
        note it differs from (l,k)
    
    theta
        Sampling points on the elevation

    Returns
    -------
        Matrix for weight and normalization 
    
    """
    return jnp.sin(theta/2)**deg_order[1]*jnp.cos(theta/2)**deg_order[2]
vmap_weight = vmap(vmap(weight, (None, 0)), (0,None))   


def wigner_d_func_recurrence(theta:jnp.ndarray, 
                             deg_order:Tuple, 
                             idx_rec:jnp.ndarray, 
                             idx_deg_gr_one:jnp.ndarray):
    """
    Function to construct a matrix from Wigner (small) d function

    Parameters
    ----------
    deg_order
        Combination of degree and order for Wigner d-functions (s, mu,vu)
        note it differs from (l,k)
    
    theta
        Sampling points on the elevation
    
    idx_rec
        Construct index for recurrence relation

    idx_deg_gr_one
        Index where the computation of recurrence should be done, for degree > 1
        Since for degree 0 and 1 it is defined prior recurrence
    


    Returns
    -------
        A weighted and normalized matrix of Wigner (small) d functions
    
    """
    # Construct normalization matrix for Jacobi polynomials
    normalize_weight = vmap_weight(theta, deg_order)

    # Construct matrix allocation given with known first two Jacobi polynomials
    matrix_allocation = matrix_params(theta, deg_order)
    
    # Construct new degree and order for recurrence with lax
    deg_order_lax = (deg_order[0],deg_order[1],deg_order[2], idx_rec)
    
    # Construct matrix 
    partial_rec_lax = partial(wigner_d_function_recurrence_lax, deg_order=deg_order_lax)
    matrix_temp = lax.scan(partial_rec_lax, theta, idx_deg_gr_one[0])
    
    # Allocate the matrix
    matrix_allocation = matrix_allocation.at[:,idx_deg_gr_one[0]].set(jnp.array(matrix_temp[1]).T)
    return normalize_weight*matrix_allocation


@jit
def matrix_phi(phi:jnp.ndarray, k:jnp.ndarray):
    """
    Function to construct a matrix with degree and azimuth angles
    in Wigner D-functions


    """
    return jnp.exp(1j*k*phi)
vmap_matrix_phi = vmap(vmap(matrix_phi, (None, 0)), (0,None))


@jit
def basis(dmm:Tuple, chi:jnp.ndarray):
    """
    Function to construct concatenated Wigner D-function for
    Spherical Near-Field Antenna Measurements

    Parameters
    ----------
    dmm
        Tuple of arrays Wigner (small) d-functions
    
    chi 
        Polarization angles

    Returns
    -------
    Basis1
        First matrix
    Basis2
        Second matrix 
    """
    dmm_plus, dmm_min = dmm
    Basis_1 = jnp.exp(1j*chi)*dmm_plus + dmm_min*jnp.exp(-1j*chi) 
    Basis_2 = jnp.exp(1j*chi)*dmm_plus - dmm_min*jnp.exp(-1j*chi)
    #norm_Basis_1 = Basis_1/jnp.linalg.norm(Basis_1,axis = 0)
    #norm_Basis_2 = Basis_2/jnp.linalg.norm(Basis_2,axis = 0)
   
    return Basis_1, Basis_2 #norm_Basis_1, norm_Basis_1  
vmap_basis = vmap(basis,0)



def wigner_snf_matrix(ang:Tuple,deg_order:Tuple, params:Tuple):

    """
    Function to construct Matrix for Spherical Near-Field Antenna measurements

    Parameters
    -----------
    params
        Parameters that is needed to construct matrix weight, the index recurrence and location degree greater than 1
    
    ang
        Angles of spherical surface (elevation, azimuth)
    
    deg_order
        Combination of degree and order (l,k), where -l <= k <= l and l = [1,1,..B]

    Returns
    -------

    """
    
    theta, phi = ang
    chi = (jnp.arange(len(theta)) % 2)*(np.pi/2)
    
    # Params
    weight_plus, weight_min, deg_order_plus, deg_order_min, idx_rec_plus, idx_deg_gr_one_plus, idx_rec_min, idx_deg_gr_one_min = params
    
    # Construct positif Wigner (small) d function
    wdp_rec = wigner_d_func_recurrence(theta, deg_order_plus, idx_rec_plus, idx_deg_gr_one_plus)
    dmm_plus = weight_plus*wdp_rec

    # Construct negative Wigner (small) d function
    wdn_rec = wigner_d_func_recurrence(theta, deg_order_min, idx_rec_min, idx_deg_gr_one_min)
    dmm_min = weight_min*wdn_rec
  
    # Exponential term 
    phi_mat = vmap_matrix_phi(phi,deg_order[1])
   
    # Generate Wigner D for s = 1 (TE) and s = 2 (TM)
    dmm = (dmm_plus, dmm_min)
    Basis_1, Basis_2 = vmap_basis(dmm, chi)
    
    # Normalize
    norm_Basis_1 = Basis_1/jnp.linalg.norm(Basis_1, axis = 0)[jnp.newaxis,:]

    ## Generate Wigner D for s = 2 (TM)
    norm_Basis_2 = Basis_2/jnp.linalg.norm(Basis_2, axis = 0)[jnp.newaxis,:]

    
    normA = jnp.concatenate((norm_Basis_1*phi_mat   , norm_Basis_2*phi_mat   ), axis = 1)
     

    
    return normA  