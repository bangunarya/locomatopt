import numpy as np
from .basealgo import BaseAlgo
from .metric import coherence
import copy
from .vectorizing_coh import vector_coherence_sh, vector_coherence_wigner, vector_coherence_snf
from numpy import linalg as LA
from .prox import project_l1_ball


class ALM(BaseAlgo):
 

    """
    Augmented Lagrangian method method to optimize angles of the sensing matrices
    for spherical near-field 
    
    References
    ----------
    
    [1]
   
    """
    def backtrack_line_search(self, angles, grad, case, u, z, w, rho):
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
        choose_grad = {'theta': grad['theta'],
                       'phi': grad['phi']}
        
        ## Update w.r.t each angle
        choose_angles = {'theta': lambda alpha: angles['theta'] + alpha*(-grad['theta']),
                         'phi' : lambda alpha: angles['phi'] + alpha*(-grad['phi'])}
        
        ## Condition on wigner
        if self.params_mat['types'] == 'wigner':
            choose_grad.update({'chi':grad['chi']})
            choose_angles.update({'chi': lambda alpha: angles['chi'] + alpha*(-grad['chi'])})
            
       
        
        ## Calculate condition 2
        ## f(x) + c * alpha * f_grad(x) * (-f_grad(x)) 
       
        cond2 = lambda alpha: (rho/2)*LA.norm(z - w)**2 + c*alpha*(choose_grad[case].T@ (-choose_grad[case])) 
        
        ## Calculate condition 1
        ## f( x + alpha * (-f_grad(x)) )
     
        ## Get new angle
        angles_temp[case] = copy.deepcopy(choose_angles[case](alpha))
        vect_coh = self.vector_coherence(self.gen_matrix(angles = angles_temp))
        cond1 = (rho/2)*LA.norm(z - vect_coh + u)**2 
         
        # Armijo condition
        #while f( x + alpha * (-f_grad(x)) ) > f(x) + c * alpha * f_grad(x) * (-f_grad(x)) :
        while (cond1 - cond2(alpha)) >= 1e-4: 
       
            alpha *= rho
            
            ## Calculate condition 1
            ## f( x + alpha * (-f_grad(x)) )
            
            angles_temp[case] = copy.deepcopy(choose_angles[case](alpha))
            vect_coh = self.vector_coherence(self.gen_matrix(angles = angles_temp))
            cond1= (rho/2)*LA.norm(z - vect_coh + u)**2 
            
           
               

        return alpha 
 
    def backtrack_line_search_prox(self, grad, z, w, rho):
        ## Using Armijo condition
        ## f( x + alpha * (-f_grad(x)) ) < f(x) + c * alpha * f_grad(x) * (-f_grad(x)) 
        ## In this case cond1 < cond2
        # init data 0 < c < 0.5 (typical:10^-4 0) < rho <= 1
        alpha = 1
        rho = 0.8
        c = 1e-4
        
        ## Calculate condition 2
        ## f(x) + c * alpha * f_grad(x) * (-f_grad(x)) 
        
        cond2 = lambda alpha : ((rho/2)*LA.norm(z - w)**2  +
                                c*alpha*grad.T@(-grad))
             
        
        ## Calculate condition 1
        ## f( x + alpha * (-f_grad(x)) )
        cond1 =  lambda alpha : ((rho/2)*LA.norm(z + alpha*(-grad) - w)**2 )
         
            
        # Armijo condition
        #while f( x + alpha * (-f_grad(x)) ) > f(x) + c * alpha * f_grad(x) * (-f_grad(x)) :
        #print(cond1, cond2)

        while cond1(alpha) > cond2(alpha): 
            alpha *= rho
       
        
#            cond1 = (rho/2)*LA.norm(x + alpha*(-grad) - w)**2
         
        
            ## Calculate condition 2
            ## f(x) + c * alpha * f_grad(x) * (-f_grad(x)) 
#            cond2 = (rho/2)*LA.norm(x - w)**2  + c*alpha*grad.T@(-grad)
         

        return alpha 
    

    def grad_angle(self, grad, z, w, rho):
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
  
        ## Calculate gradient

        gr_theta = -grad.gr_theta_total@(rho*(z - w )) 
        gr_phi = -grad.gr_phi_total@(rho*(z - w ))
        
        ## Create Dictionary
        gr_new = {'theta':gr_theta,
                  'phi': gr_phi}
        
        ## If we calculate Wigner
        if self.params_mat['types'] == 'wigner':
            
            ## Calculate gradient
            gr_chi = -grad.gr_chi_total@(rho*(z - w))
            
            ## Create Dictionary
            gr_new.update({'chi': gr_chi}) 
        
        return gr_new
    
    
    def update_prox(self, w, z, rho, step_size):
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
    
        paramLambda = 1
        stopThr = 1e-6
        grad = rho*(z - w)  
        
        if step_size == None:
            step_size = self.backtrack_line_search_prox(grad, z, w, rho)
        
        ## Gradient for the smooth
        vV = z - step_size*(grad)
        ## Proximal method to project into l1
        vX = vV - (step_size*paramLambda*project_l1_ball(vV/(paramLambda*step_size), 1, stopThr))
         
                
        return vX
 
    def update_ang(self, step_size, grad,
                   w , z ,u, rho, angles): 
    
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
       
        grad_new =  self.grad_angle(grad = grad, z = z, 
                                    w = w, rho = rho)
        if step_size == None:

            step_size = self.backtrack_line_search(angles, grad_new, 'theta', u, z, w, rho)
        ########################################### Update theta
        angles['theta'] = angles['theta'] - step_size*(grad_new['theta'])
       
        
        ########################################### Update phi
        if step_size == None:
            step_size = self.backtrack_line_search(angles, grad_new, 'phi', u, z, w, rho)
        angles['phi']   = angles['phi']  - step_size*(grad_new['phi'])

        #Update
        #angles['theta'] = copy.deepcopy(theta)
        #angles['phi'] = copy.deepcopy(phi)
        ########################################### Update chi
        if self.params_mat['types'] == 'wigner':
            
            # Generate matrix and update gradient
            if step_size == None:
                step_size = self.backtrack_line_search(angles, grad_new, 'chi', u, z, w, rho)
            angles['chi']   = angles['chi']    - step_size*grad_new['chi'] 
           # angles['chi'] = copy.deepcopy(chi)
        return angles
    
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
    
    def step_update(self, mat, angles, rho,
                    grad, vect_coh, u, z,step_size):
       
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
        #Auxellary variable
        w = vect_coh - u 
        
        ## Proximal inf norm
     
        z = self.update_prox(w, z, rho, 
                             step_size)

        ## Update for fix or all
        ## Backtracking
       
        #step_size_ang = self.backtracking_angle(grad, u, z, 
        #                                    w, rho, angles)    
         
        angles = self.update_ang(step_size, grad, w, 
                                 z , u, rho, angles)
        
        ## Get matrix  
        mat = self.gen_matrix(angles = angles)
        
        ## Get gradient
        grad = self.gen_grad(mat = mat) 
        
        ## Update coherence vector
        vect_coh = self.vector_coherence(mat = mat)
        
        ## Update mu
        u = u + rho*(z - vect_coh)
        
        return mat, angles, grad, vect_coh, u, z
    
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
        
        ## Get matrix
        mat = self.gen_matrix(angles = angles)
        
        ## Get gradient
        grad = self.gen_grad(mat = mat)
        
        ## Vectorize the product
        vect_coh = self.vector_coherence(mat = mat)
            
        ### Initial coherence
        coh = coherence(mat.normA)
    
        
        ## Dual
        u = np.zeros_like(vect_coh)
        
        ## Aux variables
        z = np.zeros_like(u)
        
        rho = 1
        
        return mat, grad, vect_coh, coh, u, z, rho
        
    
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
        ## Lower bound
        lower_bound = self.lower_bound(angles = angles)
      
        ## Initial iteration
        iterate = 0
        
        ## Initial parameter ALM
        mat, grad, vect_coh, coh, u, z, rho = self.params_alm(angles = angles)
     
        ## Initial angle
        alm_ang = copy.deepcopy(angles)
        
        while (iterate < self.params_grad['max_iter'] and 
               np.abs(coh - lower_bound) > self.params_grad['eps']):
            
            ## Add iteration
            iterate += 1
           
            
            ## Update for fix or all
            mat, angles, grad, vect_coh, u, z = self.step_update(mat, angles, rho,
                                                                 grad, vect_coh, 
                                                                 u, z, step_size)
 
            ### Store if we have better coherence
            if coherence(mat.normA) < coh:
                ## Calculate the coherence
                coh = coherence(mat.normA)
                ## Get the angles
                alm_ang = copy.deepcopy(angles)
             
        return {'coherence': coh,
                'angle': alm_ang}
