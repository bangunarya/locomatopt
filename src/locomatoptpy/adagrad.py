import numpy as np
from .basealgo import BaseAlgo
from .metric import coherence
import copy


class AdaGrad(BaseAlgo):
    
    """
    Adagrad method to optimize angles of the sensing matrices
    
    
    References
    ----------
    [1] Adaptive Subgradient Methods for Online Learning and Stochastic Optimization

        John Duchi, Elad Hazan, Yoram Singer
    """
    
    
    def params_adagrad(self, angles):
        """
        Parameters for adagrad algorithms
        Arguments:
            - angles : dict
                angle to construct matrix
                
        Returns:
            adagrad parameters : dict
        """
        histogradtheta = np.zeros(len(angles['theta'])) # historical gradients
        histogradphi = np.zeros_like(histogradtheta) # historical gradients
        histogradchi = np.zeros_like(histogradphi) # historical gradients
        
        return {'histogradtheta':histogradtheta, 'histogradphi': histogradphi,
                'histogradchi':histogradchi}
    
    def step_update(self, step_size, angles, params_adagrad, grad):
        
        """
        Perform single step update for adagrad
        Arguments:
            - step_size : float
                step size for gradient descent method
                
            - angles : dict
                parameters to optimize, angles  (theta, phi, chi)
            - params_adagrad: dict
                parameter for adagrad       
            - grad : dict
                gradient with respect to angles  (theta, phi, chi)
            - iterate : int
                current iteration
        Returns:
            - angles : dict
                updated angles
            - params_adagrad : dict
                updated adagrad parameters
        """
        
        ###################### Update Theta ####################
        # Update historical gradients
        params_adagrad['histogradtheta'] = params_adagrad['histogradtheta'] + grad.gr_theta**2
            
        ## Update decision variables
        angles['theta'] = (angles['theta'] - step_size*grad.gr_theta/
                           (np.sqrt(params_adagrad['histogradtheta']) + self.params_grad['eps']))
        
        if self.params_grad['update'] == 'fix_theta':
        
            ## Fix theta for checking the bound
            angles['theta'] = np.arccos(np.linspace(-1,1,len(angles['theta'])))
        
        ####################### Update Phi ####################
        # Update historical gradients
        params_adagrad['histogradphi'] = params_adagrad['histogradphi'] + grad.gr_phi**2
            
        ## Update decision variables
        angles['phi'] = (angles['phi'] - step_size*grad.gr_phi/
                         (np.sqrt(params_adagrad['histogradphi']) + self.params_grad['eps']))
      
        
        if self.params_mat['types'] == 'wigner':
            
            ####################### Update Chi ####################
            # Update historical gradients
            params_adagrad['histogradchi'] = params_adagrad['histogradchi'] + grad.gr_chi**2
            
            ## Update decision variables
            angles['chi'] = (angles['chi'] - step_size*grad.gr_chi/
                             (np.sqrt(params_adagrad['histogradchi']) + self.params_grad['eps']))
                             
        return angles, params_adagrad
        
    def run_algo(self, angles):
         
        """
        Perform adagrad for certain iteration.
        Arguments:
       
            - angles
                parameters to optimize, angles  (theta, phi, chi)
        Returns:
            - angles
                updated angles
            - coherence
                updated coherence 
        """
        ## AdaGrad parameter initialization
        params_adagrad = self.params_adagrad(angles = angles)
        
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
        adagrad_ang = copy.deepcopy(angles)
       
        while (iterate < self.params_grad['max_iter'] and 
               np.abs(coh - lower_bound) > self.params_grad['eps']):
            
            ## Add iteration
            iterate += 1
            
            ## Update for fix or all
            angles, params_adagrad = self.step_update(step_size = 0.05, 
                                                      angles = angles,
                                                      params_adagrad = params_adagrad,
                                                      grad = grad)
    
            ## Get matrix
            mat = self.gen_matrix(angles = angles)
            ## Get gradient
            grad = self.gen_grad(mat = mat)
             
            ### Store if we have better coherence
            if coherence(mat.normA) < coh:
            
                coh = coherence(mat.normA)
                adagrad_ang = copy.deepcopy(angles)
        
        return {'coherence': coh,
                'angle': adagrad_ang}
        