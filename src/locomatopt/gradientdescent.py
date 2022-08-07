import numpy as np
from .basealgo import BaseAlgo
from .metric import coherence
import copy
from .vectorizing_coh import matrix_coherence
from .backtracking_line import backtrack_line_search
from functools import partial


class GradDescent(BaseAlgo):

    """
    Gradient descent method to optimize angles of the sensing matrices
    
    """
 
    def obj_function(self, angles):
        matrix = self.gen_matrix(angles)

        return np.linalg.norm(matrix_coherence(matrix), self.params_grad['p_norm'])
    
    def first_condition_backtrack(self, angles, case, grad, alpha):
        angles_temp = copy.deepcopy(angles)
        angles_temp[case] = angles_temp[case] - alpha*grad[case]
        return self.obj_function(angles_temp)

    def second_condition_backtrack(self, angles, case, grad, c_alpha):
        
        return self.obj_function(angles) + c_alpha*(grad[case].conj().T @ -grad[case])
        
    def fix_theta(self, angles, grad, step_size):

        """
        Perform gradient update with fix parameter theta.
        Since we have lower bound se references

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
 
        # Update phi
      
        step_size_phi = (step_size or 
                         backtrack_line_search(
                            partial(self.first_condition_backtrack,
                                    angles, 'phi', grad),
                            partial(self.second_condition_backtrack,
                                    angles, 'phi', grad)))
         
        angles['phi'] = angles['phi'] - step_size_phi*grad['phi']

        if self.params_mat['types'] == 'wigner':
            
            # Update chi 
            step_size_chi = (step_size or 
                             backtrack_line_search(
                                partial(self.first_condition_backtrack,
                                        angles, 'chi', grad),
                                partial(self.second_condition_backtrack,
                                        angles, 'chi', grad)))

            angles['chi'] = angles['chi'] - step_size_chi*grad['chi']
        
        return angles

    def update_all(self, angles, grad, step_size):
        """
        Perform gradient update for all parameters

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
    
        # Update theta
        step_size_theta = (step_size or
                           backtrack_line_search(
                            partial(self.first_condition_backtrack, 
                                    angles, 'theta', grad),
                            partial(self.second_condition_backtrack, 
                                    angles, 'theta', grad)))
      
        angles['theta'] = angles['theta'] - step_size_theta*grad['theta']

       
        # Update phi
        step_size_phi = (step_size or 
                         backtrack_line_search(
                            partial(self.first_condition_backtrack,
                                    angles, 'phi', grad),
                            partial(self.second_condition_backtrack,
                                    angles, 'phi', grad)))
        
        angles['phi'] = angles['phi'] - step_size_phi*grad['phi']

        if self.params_mat['types'] == 'wigner':
            
            # Update chi 
            step_size_chi = (step_size or 
                             backtrack_line_search(
                                partial(self.first_condition_backtrack,
                                        angles, 'chi', grad),
                                partial(self.second_condition_backtrack,
                                        angles, 'chi', grad)))

            angles['chi'] = angles['chi'] - step_size_chi*grad['chi']
        
        return angles
    
    def step_update(self, angles, grad, step_size):
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
        
        # Dictionary for update 
        update_angles = {'fix_theta': self.fix_theta,
                         'update_all': self.update_all}
        # Update angles
        angles = update_angles[self.params_grad['update']](angles, grad, step_size)
       
        return angles
    
    def run_algo(self, angles, step_size):
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
        # Check condition for update
        if self.params_grad['update'] == 'fix_theta':
            # Fix theta for checking the bound
            angles['theta'] = np.arccos(np.linspace(-1, 1, len(angles['theta'])))

        # Lower bound
        lower_bound = self.lower_bound(angles=angles)
        
        # Initial iteration
        iterate = 0
        
        # Get matrix
        mat = self.gen_matrix(angles=angles)
        # Get gradient
        grad = self.gen_grad(mat=mat)
        
        # Initial coherence
        coh = coherence(mat.normA)
        
        # Initial angle
        graddes_ang = copy.deepcopy(angles)
        
        while (iterate < self.params_grad['max_iter'] and 
               np.abs(coh - lower_bound) > self.params_grad['eps']):
            
            # Add iteration
            iterate += 1
            
            # Update for fix or all
           
            angles = self.step_update(angles=angles, 
                                      grad=grad,
                                      step_size=step_size)
 
            # Get matrix
            mat = self.gen_matrix(angles=angles)
            # Get gradient
            grad = self.gen_grad(mat=mat)
            
            coh_new = coherence(mat.normA)
             
            # Store if we have better coherence
            if coh_new < coh:
                # Calculate the coherence
                coh = coh_new.copy()
                # Get the angles
                graddes_ang = copy.deepcopy(angles)
        
        return {'coherence': coh,
                'angle': graddes_ang}
        
    
    
