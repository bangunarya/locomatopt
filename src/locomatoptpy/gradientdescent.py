import numpy as np
from .basealgo import BaseAlgo
from .metric import coherence
import numpy.linalg as LA
import copy
from .vectorizing_coh import vector_coherence_sh, vector_coherence_wigner, vector_coherence_snf
class GradDescent(BaseAlgo):
    """
    Gradient descent method to optimize angles of the sensing matrices
    
    """
    def backtrack_line_search(self, angles, grad, case):
        ## Using Armijo condition
        ## f( x + alpha * (-f_grad(x)) ) < f(x) + c * alpha * f_grad(x) * (-f_grad(x)) 
        ## In this case cond1 < cond2
        # init data 0 < c < 0.5 (typical:10^-4 0) < rho <= 1
        alpha = 1
        rho = 0.8
        c = 1e-4 
        
        ## Temp Angles
        angles_temp = copy.deepcopy(angles)
        
        ## Gradient w.r.t each angle
        choose_grad = {'theta': grad.gr_theta,
                       'phi': grad.gr_phi}
        
        ## Update w.r.t each angle
        choose_angles = {'theta': lambda alpha: angles['theta'] + alpha*(-choose_grad['theta']),
                         'phi' : lambda alpha: angles['phi'] + alpha*(-choose_grad['phi'])}
        
        ## Condition on wigner
        if self.params_mat['types'] == 'wigner':
            choose_grad.update({'chi':grad.gr_chi})
            choose_angles.update({'chi': lambda alpha: angles['chi'] + alpha*(-choose_grad['chi'])})
            
       
        
        ## Calculate condition 2
        ## f(x) + c * alpha * f_grad(x) * (-f_grad(x)) 
        func = LA.norm(self.vector_coherence(mat=self.gen_matrix(angles=angles)),self.params_grad['p_norm'])
        cond2 = lambda alpha: func + c*alpha*(choose_grad[case].T@ (-choose_grad[case])) 
        
        ## Calculate condition 1
        ## f( x + alpha * (-f_grad(x)) )
     
        ## Get new angle
        angles_temp[case] = copy.deepcopy(choose_angles[case](alpha))
        cond1 = LA.norm(self.vector_coherence(self.gen_matrix(angles = angles_temp)),self.params_grad['p_norm'])
        
         
        # Armijo condition
        #while f( x + alpha * (-f_grad(x)) ) > f(x) + c * alpha * f_grad(x) * (-f_grad(x)) :
        while (cond1 - cond2(alpha)) >= 1e-4: 
             
            alpha *= rho
            
            ## Calculate condition 1
            ## f( x + alpha * (-f_grad(x)) )
            
            angles_temp[case] = copy.deepcopy(choose_angles[case](alpha))
            cond1 = LA.norm(self.vector_coherence(self.gen_matrix(angles = angles_temp)),self.params_grad['p_norm'])
            
           
               

        return alpha 

    def vector_coherence(self, mat):
        """
        Perform vectorization of the coherence, see paper for explanation
        
        Arguments:
          
            - mat : dict
                constructed matrix
             
        Returns:
            - vect_coh : array
                vectorizing the coherence
        """
        
        ## Matrix
        select_mat = {'sh'    : vector_coherence_sh,
                      'wigner': vector_coherence_wigner,
                      'snf'   : vector_coherence_snf}

        vect_coh = select_mat[self.params_mat['types']](params_mat = self.params_mat, 
                                                        mat = mat)
        
        return vect_coh

    def step_update(self, angles, grad,step_size):
        """
        Perform single step update for gradient descent.
        Arguments:
            - step_size : float
                step size for gradient descent method
                
            - angles
                parameters to optimize, angles  (theta, phi, chi)
             
            - grad
                gradient with respect to angles  (theta, phi, chi)
        Returns:
            - angles
                updated angles
        """
        
        ## Update theta
        if step_size == None:
            step_size = self.backtrack_line_search(angles, grad, 'theta')
        
        angles['theta'] = angles['theta'] - step_size*grad.gr_theta
            
        if self.params_grad['update'] == 'fix_theta':
        
            ## Fix theta for checking the bound
            angles['theta'] = np.arccos(np.linspace(-1,1,len(angles['theta'])))
        
    
        ## Update phi
        if step_size == None:
            step_size = self.backtrack_line_search(angles, grad, 'phi')  

        angles['phi'] = angles['phi']  - step_size*grad.gr_phi
        
       
        if self.params_mat['types'] == 'wigner':
        
            ## Update chi
            if step_size == None:
                step_size = self.backtrack_line_search(angles, grad, 'chi')
            angles['chi']   = angles['chi']   - step_size*grad.gr_chi
        
        
        return angles
    
    
    def run_algo(self,angles,step_size):
        """
        Perform gradient descent for certain iteration.
        Arguments:
       
            - angles
                parameters to optimize, angles  (theta, phi, chi)
        Returns:
            - angles
                updated angles
            - coherence
                updated coherence 
        """
        ## Lower bound
        lower_bound = self.lower_bound(angles = angles)
        
       
        ## Initial iteration
        iterate = 0
        
        ## Get matrix
        mat = self.gen_matrix(angles = angles)
        ## Get gradient
        grad = self.gen_grad(mat = mat)
        
        
        ## Initial coherence
        coh = coherence(mat.normA)
        
        ## Initial angle
        graddes_ang = copy.deepcopy(angles)
        
        while (iterate < self.params_grad['max_iter'] and 
               np.abs(coh - lower_bound) > self.params_grad['eps']):
            
            ## Add iteration
            iterate += 1
            
            ## Update for fix or all
           
            angles = self.step_update(angles = angles, 
                                      grad = grad,
                                      step_size=step_size)
 
            ## Get matrix
            mat = self.gen_matrix(angles = angles)
            ## Get gradient
            grad = self.gen_grad(mat = mat)
             
            ### Store if we have better coherence
            if coherence(mat.normA) < coh:
                ## Calculate the coherence
                coh = coherence(mat.normA)
                ## Get the angles
                graddes_ang = copy.deepcopy(angles)
        
        return {'coherence': coh,
                'angle': graddes_ang}
        
    
    