import numpy as np
from .basealgo import BaseAlgo
from .metric import coherence
import copy

class AdaDelta(BaseAlgo):
    """
    Adadelta method to optimize angles of the sensing matrices
    
    
    References
    ----------
    [1] Adadelta: an adaptive learning rate method
        MD Zeiler
    """
    
    def params_adadelta(self, angles):
        
        """
        Parameters for adadelta algorithms
        Arguments:
            - angles : dict
                angle to construct matrix
                
        Returns:
            adadelta parameters : dict
        """
        
        beta = 0.95 ## Ada Delta
                    
        acculGradtheta = np.zeros(len(angles['theta'])) # accumulated gradients
        acculDeltatheta = np.zeros(len(angles['theta'])) # accumulated updates 
        
        acculGradphi = np.zeros_like(acculGradtheta)
        acculDeltaphi = np.zeros_like(acculDeltatheta)
                    
        
        acculGradchi = np.zeros_like(acculGradtheta)
        acculDeltachi = np.zeros_like(acculDeltatheta)
                    
        return {'acculGradtheta':acculGradtheta, 'acculDeltatheta':acculDeltatheta,
                'acculGradphi':acculGradphi, 'acculDeltaphi':acculDeltaphi,
                'acculGradchi':acculGradchi, 'acculDeltachi':acculDeltachi,
                'beta': beta}
        
      
    def step_update(self, angles,params_adadelta, grad):
        
        """
        Perform single step update for adadelta
        Arguments:
             
            - angles : dict
                parameters to optimize, angles  (theta, phi, chi)
            - params_adadelta: dict
                parameter for adadelta     
            - grad : dict
                gradient with respect to angles  (theta, phi, chi)
           
        Returns:
            - angles : dict
                updated angles
            - params_adadelta : dict
                updated adadelta parameters
        """
         
        ######################## Update theta
        ## Update accumulated gradients
        params_adadelta['acculGradtheta'] = (params_adadelta['beta']*params_adadelta['acculGradtheta'] +
                                             (1.0 - params_adadelta['beta'])*grad.gr_theta**2)
        ## Calculate update
        dCurrent = (np.sqrt(params_adadelta['acculDeltatheta'] + self.params_grad['eps'])/
                    np.sqrt(params_adadelta['acculGradtheta'] + self.params_grad['eps']))*grad.gr_theta

        ## Update accumulated updates
        params_adadelta['acculDeltatheta']=(params_adadelta['beta']*params_adadelta['acculDeltatheta'] + 
                                            (1.0 - params_adadelta['beta'])*dCurrent**2)
            
        ## Update decision variables
        angles['theta'] = angles['theta']  - dCurrent
        
        if self.params_grad['update'] == 'fix_theta':
        
            ## Fix theta for checking the bound
            angles['theta'] = np.arccos(np.linspace(-1,1,len(angles['theta'])))
        
        ####################### Update Phi ####################
        ## Update accumulated gradients
        params_adadelta['acculGradphi'] = (params_adadelta['beta']*params_adadelta['acculGradphi'] +
                                           (1.0 - params_adadelta['beta'])*grad.gr_phi**2)
        ## Calculate update
        dCurrent = (np.sqrt(params_adadelta['acculDeltaphi'] + self.params_grad['eps'])/
                    np.sqrt(params_adadelta['acculGradphi'] + self.params_grad['eps']))*grad.gr_phi

        ## Update accumulated updates
        params_adadelta['acculDeltaphi'] = (params_adadelta['beta']*params_adadelta['acculDeltaphi'] + 
                                            (1.0 - params_adadelta['beta'])*dCurrent**2)
            
        ## Update decision variables
        angles['phi'] = angles['phi']  - dCurrent
 
        
        if self.params_mat['types'] == 'wigner':
        ######################## Update Chi #######################
         
            ## Update accumulated gradients
            params_adadelta['acculGradchi'] = (params_adadelta['beta']*params_adadelta['acculGradchi'] +
                                               (1.0 - params_adadelta['beta'])*grad.gr_chi**2)
            ## Calculate update
            dCurrent = (np.sqrt(params_adadelta['acculDeltachi'] + self.params_grad['eps'])/
                        np.sqrt(params_adadelta['acculGradchi'] + self.params_grad['eps']))*grad.gr_chi

            ## Update accumulated updates
            params_adadelta['acculDeltachi'] = (params_adadelta['beta']*params_adadelta['acculDeltachi'] +
                                                (1.0 - params_adadelta['beta'])*dCurrent**2)
            
            ## Update decision variables
            angles['chi'] = angles['chi']  - dCurrent
       
        
        return angles, params_adadelta
    
    
    
    def run_algo(self, angles):
        """
        Perform adadelta for certain iteration.
        Arguments:
       
            - angles
                parameters to optimize, angles  (theta, phi, chi)
        Returns:
            - angles
                updated angles
            - coherence
                updated coherence 
        """
       
        ## params adadelta
        params_adadelta = self.params_adadelta(angles = angles)
        
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
        adadelta_ang = copy.deepcopy(angles)
       
        while (iterate < self.params_grad['max_iter'] and 
               np.abs(coh - lower_bound) > self.params_grad['eps']):
            
            ## Add iteration
            iterate += 1
            
            
            ## Update for fix or all
            angles, params_adadelta = self.step_update(angles = angles,
                                                       params_adadelta = params_adadelta,
                                                       grad = grad)
      
            ## Get matrix
            mat = self.gen_matrix(angles = angles)
        
            ## Get gradient
            grad = self.gen_grad(mat = mat)
     
            ### Store if we have better coherence
            if coherence(mat.normA) < coh:
           
                coh = coherence(mat.normA)
                adadelta_ang = copy.deepcopy(angles)
                
        
        return {'coherence': coh,
                'angle': adadelta_ang}
        