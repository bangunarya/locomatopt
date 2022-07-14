import numpy as np
from .basealgo import BaseAlgo
from .metric import coherence
import copy

class Adam(BaseAlgo):
    
    """
    Adam method to optimize angles of the sensing matrices
    
    
    References
    ----------
    [1] Adam: A method for stochastic optimization
        DP Kingma, J Ba
    """
    
   
    def params_adam(self, angles):
        """
        Parameters for adam algorithms
        Arguments:
            - angles : dict
                angle to construct matrix
                
        Returns:
            Adam parameters : dict
        """
        beta1 = 0.9 ## ADAM parameter
        beta2 = 0.999 ## ADAM parameter
        
        ## ADAM parameters initialization
        mm_phi = np.zeros(len(angles['theta']))
        v_phi = np.zeros(len(angles['theta']))
        
        mm_chi = np.zeros_like(mm_phi)
        v_chi = np.zeros_like(v_phi)
        
        mm_theta = np.zeros_like(mm_phi)
        v_theta = np.zeros_like(v_phi)
        
        return {'v_theta': v_theta,'v_phi': v_phi,'v_chi': v_chi,
                'mm_theta': mm_theta, 'mm_phi': mm_phi, 'mm_chi': mm_chi,
                'beta1': beta1, 'beta2': beta2}
    
    def step_update(self, step_size,angles, params_adam, grad, iterate):
        """
        Perform single step update for adam
        Arguments:
            - step_size : float
                step size for gradient descent method
                
            - angles : dict
                parameters to optimize, angles  (theta, phi, chi)
            - params_adam : dict
                parameter for adam       
            - grad : dict
                gradient with respect to angles  (theta, phi, chi)
            - iterate : int
                current iteration
        Returns:
            - angles : dict
                updated angles
            - params_adam : dict
                updated adam parameters
        """
        
        ########################## Update for theta ######################
        
        ## Update biased 1st moment estimate
        params_adam['mm_theta'] = (params_adam['beta1']*params_adam['mm_theta'] +
                                   (1.0 - params_adam['beta1'])*grad.gr_theta)
        
        ## Update biased 2nd raw moment estimate
        params_adam['v_theta'] = (params_adam['beta2']*params_adam['v_theta'] + 
                                  (1.0 - params_adam['beta2'])*grad.gr_theta**2)
        

        ## Compute bias-corrected 1st moment estimate
        mHat_theta = params_adam['mm_theta']/(1.0 - (params_adam['beta1']**iterate))
        
        ## Compute bias-corrected 2nd raw moment estimate
        vHat_theta = params_adam['v_theta']/(1.0 - (params_adam['beta2']**iterate))
        
        ## Update angles theta
        angles['theta'] = (angles['theta'] - step_size*
                           mHat_theta/(np.sqrt(vHat_theta) + self.params_grad['eps']))
        
        
        if self.params_grad['update'] == 'fix_theta':
        
            ## Fix theta for checking the bound
            angles['theta'] = np.arccos(np.linspace(-1,1,len(angles['theta'])))
   
        ########################## Update for phi ######################
        
        ## Update biased 1st moment estimate
        params_adam['mm_phi'] = (params_adam['beta1']*params_adam['mm_phi'] + 
                                 (1.0 - params_adam['beta1'])*grad.gr_phi)
        
        ## Update biased 2nd raw moment estimate
        params_adam['v_phi'] = (params_adam['beta2']*params_adam['v_phi'] + 
                                (1.0 - params_adam['beta2'])*grad.gr_phi**2)
        

        ## Compute bias-corrected 1st moment estimate
        mHat_phi = params_adam['mm_phi']/(1.0 - (params_adam['beta1']**iterate))
        
        ## Compute bias-corrected 2nd raw moment estimate
        vHat_phi = params_adam['v_phi']/(1.0 - (params_adam['beta2']**iterate))
        
        ## Update angles phi
        angles['phi'] = (angles['phi'] - step_size*
                         vHat_phi/(np.sqrt(vHat_phi) + self.params_grad['eps']))
        
        
        
        
        ## Check which matrix
        if self.params_mat['types'] == 'wigner':
            ######################### Update chi ############################
           
            ## Update biased 1st moment estimate
            params_adam['mm_chi'] = (params_adam['beta1']*params_adam['mm_chi'] + 
                                 (1.0 - params_adam['beta1'])*grad.gr_chi)
        
            ## Update biased 2nd raw moment estimate
            params_adam['v_chi'] = (params_adam['beta2']*params_adam['v_chi'] + 
                                (1.0 - params_adam['beta2'])*grad.gr_chi**2)
        

            ## Compute bias-corrected 1st moment estimate
            mHat_chi = params_adam['mm_chi']/(1.0 - (params_adam['beta1']**iterate))
        
            ## Compute bias-corrected 2nd raw moment estimate
            vHat_chi = params_adam['v_chi']/(1.0 - (params_adam['beta2']**iterate))
        
            ## Update angles chi
            angles['chi'] = (angles['chi'] - step_size*
                   mHat_chi/(np.sqrt(vHat_chi) + self.params_grad['eps']))
            
            
       
        
        return angles, params_adam
    
    def run_algo(self, angles):
        """
        Perform adam for certain iteration.
        Arguments:
       
            - angles
                parameters to optimize, angles  (theta, phi, chi)
        Returns:
            - angles
                updated angles
            - coherence
                updated coherence 
        """
      
        
        ## ADAM parameters initialization 
        
        params_adam = self.params_adam(angles = angles)
        
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
        adam_ang = copy.deepcopy(angles)
        
        while (iterate < self.params_grad['max_iter'] and 
               np.abs(coh - lower_bound) > self.params_grad['eps']):
            
            ## Add iteration
            iterate += 1
           
            
            ## Update for fix or all
    
            angles, params_adam = self.step_update(step_size = 0.05, angles = angles, 
                                                   params_adam = params_adam, grad = grad, 
                                                   iterate = iterate)
           
            ## Get matrix
            mat = self.gen_matrix(angles = angles)
            ## Get gradient
            grad = self.gen_grad(mat = mat)
             
            ### Store if we have better coherence
            if coherence(mat.normA) < coh:
                ## Calculate the coherence
                coh = coherence(mat.normA)
                ## Get the angles
                adam_ang = copy.deepcopy(angles)
        
        return {'coherence': coh,
                'angle': adam_ang}
        
    
    
    
     