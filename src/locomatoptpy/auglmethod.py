import numpy as np
from .basealgo import BaseAlgo
from .metric import coherence
import copy
from .vectorizing_coh import matrix_coherence
from .prox import project_l1_ball
from .backtracking_line import backtrack_line_search
from functools import partial


class ALM(BaseAlgo):
 
    """
    Augmented Lagrangian method method to optimize angles of the sensing matrices
    for spherical near-field 
    
    References
    ----------
    
    [1]
   
    """
    def obj_function(self, params):

        return np.linalg.norm(params, 2)**2

    def first_condition_backtrack_prox(self, z_aux, vect_coh, u_dual,
                                       grad, rho, alpha):
        return (rho/2)*self.obj_function(z_aux + alpha*(-grad) - (vect_coh - u_dual))

    def second_condition_backtrack_prox(self, z_aux, vect_coh, u_dual,
                                        grad, rho, c_alpha):
        return (rho/2)*self.obj_function(z_aux - (vect_coh - u_dual)) + c_alpha*grad.T@-grad

    def first_condition_backtrack_ang(self, angles, case, grad, 
                                      u_dual, z_aux, rho, alpha):
        angles_temp = copy.deepcopy(angles)
        print(alpha)
 
         
        angles_temp[case] = angles_temp[case] - alpha*grad[case]

        return (rho/2)*self.obj_function(z_aux - (matrix_coherence(self.gen_matrix(angles_temp)) - 
                                         u_dual))
    
    def second_condition_backtrack_ang(self, angles, case, grad,
                                       u_dual, z_aux, rho, c_alpha):   
        return (rho/2)*(self.obj_function(z_aux - (matrix_coherence(self.gen_matrix(angles)) - 
                                          u_dual)) + c_alpha*(grad[case].T @ -grad[case]))
    
    def grad_angle(self, grad, z_aux, vect_coh, u_dual, rho):
        """
        Perform gradient of Lagrangian with respect to angle
        see the paper
        Arguments:
 
            
            - grad : dict
                gradient with respect to angles  (theta, phi, chi)
            - z,w : array
                Augmented Lagrangian variables, see the paper
                
            - rho : float
                Augmented Lagrangian variables, see the paper
             
        Returns:
            - grad_new : dict
                gradient
        """
  
        # Calculate gradient

        gr_theta = -grad['theta_func']@(rho*(z_aux - (vect_coh - u_dual)))
        gr_phi = -grad['phi_func']@(rho*(z_aux - (vect_coh - u_dual)))
        
        # Create Dictionary
        gr_new = {'theta': gr_theta,
                  'phi': gr_phi}
        
        # If we calculate Wigner
        if self.params_mat['types'] == 'wigner':
            # Calculate gradient
            gr_chi = -grad['chi_func']@(rho*(z_aux - (vect_coh - u_dual)))
            # Create Dictionary
            gr_new.update({'chi': gr_chi})
        
        return gr_new

    def update_prox(self, vect_coh, u_dual, 
                    z_aux, rho, step_size):
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
    
        paramLambda = 1.0
        stopThr = 1e-6
        grad = rho*(z_aux - (vect_coh - u_dual))
        
        if step_size is None:
            step_size = backtrack_line_search(partial(self.first_condition_backtrack_prox,
                                                      z_aux, vect_coh,
                                                      u_dual, grad, rho),
                                              partial(self.second_condition_backtrack_prox,
                                                      z_aux, vect_coh,
                                                      u_dual, grad, rho))                                    
 
        # Gradient for the smooth
        vV = z_aux - step_size*grad
        # Proximal method to project into l1
        vX = vV - (step_size*paramLambda*project_l1_ball(vV/(paramLambda*step_size), 1, stopThr))
                     
        return vX

    def fix_theta(self, angles, grad, step_size,
                  u_dual, z_aux, rho):

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
      
        # Calculate new gradient
        
        # grad = self.grad_angle(grad=self.gen_grad(mat=self.gen_matrix(angles=angles)), 
        #                       z_aux=z_aux, vect_coh=matrix_coherence(self.gen_matrix(angles)), 
        #                       u_dual=u_dual, rho=rho)
        # Update phi
        step_size_phi = (step_size or backtrack_line_search(
                                        partial(self.first_condition_backtrack_ang,
                                                angles, 'phi', grad, u_dual, z_aux, rho),
                                        partial(self.second_condition_backtrack_ang,
                                                angles, 'phi', grad, u_dual, z_aux, rho)))
        #print(step_size_phi)
        angles['phi'] = angles['phi'] - step_size_phi*grad['phi']
        
        if self.params_mat['types'] == 'wigner':
            # Calculate new gradient
            
            # grad = self.grad_angle(grad=self.gen_grad(mat=self.gen_matrix(angles=angles)), 
            #                       z_aux=z_aux, vect_coh=matrix_coherence(self.gen_matrix(angles)), 
            #                       u_dual=u_dual, rho=rho)
            # Update chi 
            step_size_chi = (step_size or backtrack_line_search(
                                partial(self.first_condition_backtrack_ang,
                                        angles, 'chi', grad, u_dual, z_aux, rho),
                                partial(self.second_condition_backtrack_ang,
                                        angles, 'chi', grad, u_dual, z_aux, rho)))

            angles['chi'] = angles['chi'] - step_size_chi*grad['chi']
        
        return angles

    def update_all(self, angles, grad, step_size,
                   u_dual, z_aux, rho):

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
                            partial(self.first_condition_backtrack_ang,
                                    angles, 'theta', grad, u_dual, z_aux, rho),
                            partial(self.second_condition_backtrack_ang,
                                    angles, 'theta', grad, u_dual, z_aux, rho)))

        angles['theta'] = angles['theta'] - step_size_theta*grad['theta']
        
        # Calculate new gradient
        # grad = self.grad_angle(grad=self.gen_grad(mat=self.gen_matrix(angles=angles)), 
        #                       z_aux=z_aux, vect_coh=matrix_coherence(self.gen_matrix(angles)), 
        #                       u_dual=u_dual, rho=rho)
        # Update phi
        step_size_phi = (step_size or 
                         backtrack_line_search(
                            partial(self.first_condition_backtrack_ang,
                                    angles, 'phi', grad, u_dual, z_aux, rho),
                            partial(self.second_condition_backtrack_ang,
                                    angles, 'phi', grad, u_dual, z_aux, rho)))
        angles['phi'] = angles['phi'] - step_size_phi*grad['phi']

        if self.params_mat['types'] == 'wigner':
            # Calculate new gradient
            # grad = self.grad_angle(grad=self.gen_grad(mat=self.gen_matrix(angles=angles)), 
            #                              z_aux=z_aux, 
            #                              vect_coh=matrix_coherence(self.gen_matrix(angles)), 
            #                              u_dual=u_dual, rho=rho)
            # Update chi 
            step_size_chi = (step_size or 
                             backtrack_line_search(
                                partial(self.first_condition_backtrack_ang,
                                        angles, 'chi', grad, u_dual, z_aux, rho),
                                partial(self.second_condition_backtrack_ang,
                                        angles, 'chi', grad, u_dual, z_aux, rho)))

            angles['chi'] = angles['chi'] - step_size_chi*grad['chi']
        
        return angles
 
    def update_ang(self, step_size, grad,
                   vect_coh, u_dual, z_aux, rho, angles): 
    
        """
        Perform single step to update the angles
        
        Arguments:
 
            - step_size : float
                Step size for algorithms
            - grad : dict
                gradient with respect to angles
            - angles : dict
                current angles (theta, phi, chi)
            - z,w : array
                Augmented Lagrangian variables, see the paper
                
            - rho : float
                Augmented Lagrangian variables, see the paper
             
        Returns:
            - angles : dict
                updated angles
        """
   
        grad_new = self.grad_angle(grad=grad, z_aux=z_aux,
                                   vect_coh=vect_coh,
                                   u_dual=u_dual, rho=rho)
        
        # Dictionary for update 
        update_angles = {'fix_theta': self.fix_theta,
                         'update_all': self.update_all}
        # Update angles
        angles = update_angles[self.params_grad['update']](angles, grad_new, step_size,
                                                           u_dual, z_aux, rho)
    
        return angles
    
    def step_update(self, angles, rho,
                    grad, vect_coh, u_dual,
                    z_aux,
                    step_size):
       
        """
        Perform single step update for augmented Lagrangian method
        Arguments:
            - mat : dict
                constructed matrix
 
            - angles : dict
                parameters to optimize, angles  (theta, phi, chi)
            - vect_coh ; array
                vectorization of the coherence
            - grad : dict
                gradient with respect to angles  (theta, phi, chi)
            - z,u : array
                Augmented Lagrangian variables, see the paper
            - rho : float
                Augmented Lagrangian variables, see the paper
        Returns:
            - mat : dict
                updated constructed matrix
 
            - angles : dict
                parameters to optimize, angles  (theta, phi, chi)
            - vect_coh ; array
                vectorization of the coherence
            - grad : dict
                gradient with respect to angles  (theta, phi, chi)
            - z,u : array
                Augmented Lagrangian variables, see the paper
          
        """
     
        # Proximal inf norm      
        z_aux = self.update_prox(vect_coh, u_dual, 
                                 z_aux, rho,
                                 step_size)
        
        # Update for fix or all     
        angles = self.update_ang(step_size, grad,  vect_coh, 
                                 u_dual, 
                                 z_aux, rho, angles)
         
        # Get matrix
        #print(angles)
        mat = self.gen_matrix(angles=angles)
        # Get gradient
        grad = self.gen_grad(mat) 
        
        # Update coherence vector
        vect_coh = matrix_coherence(mat)
         
        # Update mu
        u_dual = u_dual + rho*(z_aux - vect_coh)
        
        return angles, grad, vect_coh, u_dual, z_aux
    
    def params_alm(self, angles):
        
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
      
        # Get gradient
        grad = self.gen_grad(self.gen_matrix(angles=angles))
        
        # Vectorize the product
        vect_coh = matrix_coherence(self.gen_matrix(angles=angles))
            
        # Initial coherence
        coh = np.max(vect_coh)
    
        # Dual
        u_dual = np.random.randn(vect_coh.shape[0])
        
        # Aux variables
        z_aux = np.random.randn(vect_coh.shape[0])
        
        rho = 1
        
        return grad, vect_coh, coh, u_dual, z_aux, rho
        
    def run_algo(self, angles, step_size):
        
        """
        Perform augmented Lagrangian method for certain iteration.
        Arguments:
       
            - angles
                parameters to optimize, angles  (theta, phi, chi)
        Returns:
            - angles
                updated angles
            - coherence
                updated coherence 
        """
        if self.params_grad['update'] == 'fix_theta':
            # Fix theta for checking the bound
            angles['theta'] = np.arccos(np.linspace(-1, 1, len(angles['theta'])))
        # Lower bound
        lower_bound = self.lower_bound(angles=angles)
      
        # Initial iteration
        iterate = 0
        
        # Initial parameter ALM
        grad, vect_coh, coh, u_dual, z_aux, rho = self.params_alm(angles=angles)
        
        # Initial angle
        alm_ang = copy.deepcopy(angles)
        
        while (iterate < self.params_grad['max_iter'] and 
               np.abs(coh - lower_bound) > self.params_grad['eps']):
            
            # Add iteration
            iterate += 1
           
            # Update for fix or all
            angles, grad, vect_coh, u_dual, z_aux = self.step_update(angles, rho,
                                                                     grad, vect_coh, 
                                                                     u_dual, z_aux, 
                                                                     step_size)
            # Store if we have better coherence
            mat = self.gen_matrix(angles=angles)
            
            if coherence(mat.normA) < coh:
                # Calculate the coherence
                coh = coherence(mat.normA)
                # Get the angles
                alm_ang = copy.deepcopy(angles)
                
        return {'coherence': coh,
                'angle': alm_ang}
