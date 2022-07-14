import numpy as np
from .matrix import MatrixSH, MatrixWigner, MatrixSNF
from .gradient import GradSH, GradWigner, GradWignerSNF
from .metric import coherence, boundcoherence
import copy 

class BaseAlgo:
    
    """
    Base class for optimization algorithms
    """
    
    
    def __init__(self,params_mat, params_grad):
        
        '''
        Parameters
        ----------
        params_mat : dict
            Parameters to construct specific matrix with parameters bandwidth (B)
            column dimension (N), type of matrix (types), combination degree and orders (col_comb),
            spherical near-field cas or general expansion (case)
            
        params_grad : dict
            Parameters to construct gradient algorithm with maximum iteration (max_iter),
            threshold for stopping criterion (eps), update for all angles or fix theta (fix_theta)
            specific number for lp-norm (p)
            
        References
        ----------
        [1]
        [2] Sensing matrix design and sparse recovery on the sphere and the rotation group
            A Bangun, A Behboodi, R Mathar - IEEE Transactions on Signal Processing
    
        [3] Coherence bounds for sensing matrices in spherical harmonics expansion
            A Bangun, A Behboodi, R Mathar - 2018 IEEE ICASSP
    
        [4] Tight bounds on the mutual coherence of sensing matrices for Wigner D-functions 
            on regular grids
            A Bangun, A Behboodi, R Mathar - 2021 Sampling Theory, Signal Processing, and Data Analysis
        
        '''
        
        self.params_mat = params_mat
        self.params_grad = params_grad

    
    def lower_bound(self,angles):
        """
        Calculate the lower bound coherence according to update of the gradient
        Arguments:
            - angles : dict
              intial angle for gradient method
           
        Returns:
            - coherence bound
                For fixed elevation angle (theta), the lower bound is given in the references [2,3,4]
                For general matrix and update all parameters, we have Welch bound
        """
        
        lower_bound = boundcoherence(m = len(angles['theta']),
                                     N = self.params_mat['N'],
                                     B = self.params_mat['B'])
        
        return lower_bound[self.params_grad['update']] 
    
    def gen_matrix(self, angles):
        """
        Construct the matrix according to the parameters
        Arguments:
            - angles : dict
              intial angle for gradient method
           
        Returns:
            - mat : dict
              matrix spherical harmonics, Wigner D-functions, SNF
        """
        ## Matrix
        select_mat = {'sh'    : MatrixSH,
                      'wigner': MatrixWigner,
                      'snf'   : MatrixSNF}

        mat = select_mat[self.params_mat['types']](B = self.params_mat['B'],
                                                   angles = angles,
                                                   case = self.params_mat['case'])
        
        return mat
    
    def gen_grad(self, mat):
        """
        Calculate the gradient of coherence for specifict matrix
        Arguments:
            - mat : dict
              generated matrix
           
        Returns:
            - grad : dict
              gradient with respect to angles (theta, phi, chi)
        """
        ## Gradient
        select_grad = {'sh'    : GradSH,
                       'wigner': GradWigner,
                       'snf'   : GradWignerSNF}
         
        grad = select_grad[self.params_mat['types']](matrix = mat, 
                                                     col_comb = self.params_mat['col_comb'],
                                                     p = self.params_grad['p_norm'])        
    
        return grad
    
    
 
        
        
        
    