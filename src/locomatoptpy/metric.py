"""
This function contains collection of all metrics, for example to calculate coherence,
the coherence bound maximum product of same degree and orders
"""

import numpy as np
from itertools import combinations
from scipy.special import lpmv as asLeg
from numpy import linalg as LA

def params_matrix(types, B, case):
    
    """ 
    Functions to calculate the size of the column matrix and
    to generate combination of index column to calculate the coherence
    
    Parameters
    ----------
    types : str
            As selector to generate column combination (col_comb) of
            matrix spherical harmonics, Wigner D functions or spherical 
            near-field measurements
            
    B: int
        Bandlimited of the functions
        
    case: str
          To choose between normal expansion or cases for antenna measurement
          As discussed in [1], in antenna measurement the DC component l (degree) = 0
          is not considered
        
    Returns
    -------
    N: int
        Column dimension of the matrix
    col_comb : ndarray
        Combination of index column
    
    
    Reference
    ---------
    [1] Hald, Jørgen, and Frank Jensen. Spherical near-field antenna measurements. Vol. 26. Iet, 1988.

    """
    if case == 'antenna':
        dim = {'sh':B**2 - 1, 'wigner':B*(2*B-1)*(2*B+1)//3 - 1,
               'snf':2*B**2 + 4*B}
        N = dim[types]
        comb = {'sh': np.array(list(combinations(range(N),2))),
                'wigner': np.array(list(combinations(range(N),2))),
                'snf': np.array(np.meshgrid(range(N//2), range(N//2))).T.reshape(-1,2)}
        col_comb = comb[types]
        
    else:
        
        dim = {'sh':B**2, 'wigner':B*(2*B-1)*(2*B+1)//3,
               'snf':2*B**2 + 4*B}
        N = dim[types]
        
        comb = {'sh': np.array(list(combinations(range(N),2))),
                'wigner': np.array(list(combinations(range(N),2))),
                'snf': np.array(np.meshgrid(range(N//2), range(N//2))).T.reshape(-1,2)}
        col_comb = comb[types]
      
    return N, col_comb
    
    
def coherence(normA):
    
    """ 
    Functions to calculate coherence of the normalized matrix   
    
    Parameters
    ----------
    normA : ndarray
        Normalized matrix 
        
    Returns
    -------
    cohernece: float
        Coherence of the matrix
        
    """
    ## Get column dimension
    N = normA.shape[1]
    ## Calculate gram matrix
    gram = np.dot(normA.conjugate().T,normA)
    
    return np.max(abs(gram - np.identity(N)))
 

def boundcoherence(m,N,B):
    
     
    """ 
    Functions to calculate  Welch bound [1] and the coherence bound in the
    paper [2]
   

    Parameters
    ----------
    m : int
        row dimension of the matrix
    N : int
        column dimension of the matrix
    B : int
        bandlimited of the function
        
    Returns
    -------
    welch: float
        Welch bound
    
    legbound: fload
        Bound given in [2,3,4], w.r.t product of Legendre
        
    
    Reference
    ---------
    
    [1] Lower bounds on the maximum cross correlation of signals by Lloyd Welch
        IEEE Transaction on Information Theory 
        
    [2] Sensing matrix design and sparse recovery on the sphere and the rotation group
        A Bangun, A Behboodi, R Mathar - IEEE Transactions on Signal Processing
    
    [3] Coherence bounds for sensing matrices in spherical harmonics expansion
        A Bangun, A Behboodi, R Mathar - 2018 IEEE ICASSP
    
    [4] Tight bounds on the mutual coherence of sensing matrices for Wigner D-functions 
        on regular grids
        A Bangun, A Behboodi, R Mathar - 2021 Sampling Theory, Signal Processing, and Data Analysis
    
    """
    
    ## Calculate Welch bound
    welch = np.sqrt((N-m)/((N-1)*m))
    
    ## Generate fix sampling on theta. Since the bound is derived for fix theta [0,pi]
    x = np.linspace(-1,1,m)
    
    ## Generate Legendre polynomials for degree B-1 and B-3 as given in the bound
    PB1 = asLeg(0,B-1,x)
    PB3 = asLeg(0,B-3,x)
    
    ## Calculate the bound in [2]
    legbound = abs(np.inner((PB1/LA.norm(PB1)),(PB3/LA.norm(PB3))))
    return {'update_all': welch, 
            'fix_theta' : legbound}

###########################################################
## Maximum of the product same degree and orders
## 
## Input : normalized matrix, and combination degree orders
## Output : maximum product for same orders k1 = k2, n1 = n2
##
############################################################
def maxprod(normA, deg_order):
    
    """ 
    Functions to calculate maximum of the product same degree and orders
    as described in paper [1,2]
   

    Parameters
    ----------
    normA : ndarray
        Normalized matrix 
        
    deg_order : ndarray
        Degree and orders of the matrix 
    
    Returns
    -------
    max_prod : float
        Maximum product of the product same degree and orders
        
    
    Reference
    ---------
    
     
    [1] Sensing matrix design and sparse recovery on the sphere and the rotation group
        A Bangun, A Behboodi, R Mathar - IEEE Transactions on Signal Processing
        
    [2] Tight bounds on the mutual coherence of sensing matrices for Wigner D-functions on regular grids
        A Bangun, A Behboodi, R Mathar - 2021 Sampling Theory, Signal Processing, and Data Analysis
    
    """
    ## Size of column matrix
    N = normA.shape[1]
    
    ## Initialization
    max_prod = 0
    
    ## Run for all combination
    for subset in combinations(range(N),2):
        ## Get each combination
        comb_column = np.array(subset)
        
        ## calculate combination for degree and oders
        comb_lkn = [deg_order[comb_column[0],:], 
                    deg_order[comb_column[1],:]]
        
        ## Calculate difference for order k
        k = comb_lkn[0][1] - comb_lkn[1][1]
        ## Calculate difference for order n
        n = comb_lkn[0][2] - comb_lkn[1][2]
        ## Sum and check which index give 0
        knzero = abs(k) + abs(n)
        
        if knzero == 0:
            
            max_prod = max(abs(normA[:,comb_column[0]]).dot(normA[:,comb_column[1]]), max_prod)
        
    return max_prod
