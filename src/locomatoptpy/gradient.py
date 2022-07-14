 
from itertools import combinations
import numpy as np
import numpy.matlib as npmat
 

class Gradient:
    
    """ 
    Base class to calculate derivative of product spherical harmonics and Wigner D-functions.
      
    Parameters
    ----------
    matrix : Matrix class
            instance from Matrix class, consist of:
            - Matrix A, normalized matrix A
            - Matrix Legendre (Plk) or Wigner (small) d-function (wignerd)
            - Derivative of associated Legendre (dPlk) or Wigner (small) d-functions
    
    col_comb : ndarray
            col_comb contains the combination of column index of normalized matrix A
            to evaluate the coherence
        
    
    
    """

    def __init__(self, matrix, col_comb, p):
        self.matrix = matrix
        self.col_comb = col_comb
        self.p = p 
        self.grad_total()


class GradSH(Gradient):
    
    """
    Class to generate gradient of product combination columns of 
    spherical harmonics matrix. 
    Since we want to optimize the coherence of the matrix (inner product of difference columns), we have
    to calculate the gradient w.r.t theta, phi
    
    
    Attributes
    -------
    
    gr_theta, gr_phi: ndarray
        Gradient of the product of spherical harmonics 
        with respect to angles, i.e., theta, phi 
    
    
    
    """
    
    def grad_total(self):
        
        """
        Method to calculate gradient total with respect
        to theta, phi, chi
        
        """
        self.deriv_norm()
        ## Calculate derivative of theta
        self.deriv_theta()
        
        ## Calculate derivative of phi
        self.deriv_phi()
        
    
    def comb_pos(self):
        """
        Method to generate combination of degree and orders
        
        """
         
        
        ## Combination of degree and orders
        comb_lk = [self.matrix.deg_order[self.col_comb[:,0],:],
                   self.matrix.deg_order[self.col_comb[:,1],:]]
        
        ## Product of combination of degree and order associated Legendre 
        self.ProductasLeg = (self.matrix.Plk[:,self.col_comb[:,0]]*
                             self.matrix.Plk[:,self.col_comb[:,1]])
        
        ## Differences order
        self.k = comb_lk[0][:,1] - comb_lk[1][:,1]
    
    def deriv_norm(self):
        
        """
        Method to calculate derivative of norm since we 
        approximate infinity norm for p-norm for high p
        """
        
        self.comb_pos()
        
        ## Allocation
       
        
        phi = self.matrix.angles['phi']
        
        ## q-norm, q = 8
        q = self.p
        
         
        ## Derivative of norm
        
        
        self.mat_cos = np.cos(np.outer(phi,self.k))
        self.mat_sin = np.sin(np.outer(phi,self.k))
    
        self.Qnorm = (q/2.0)*np.sqrt(np.abs(np.sum(self.ProductasLeg*self.mat_cos,0))**2 +
                                     np.abs(np.sum(self.ProductasLeg*self.mat_sin,0))**2)**(q-2)
    
        self.Qnorm1 = (1/q)*np.sum(np.sqrt(np.abs(np.sum(self.ProductasLeg*self.mat_cos,0))**2 + 
                                           np.abs(np.sum(self.ProductasLeg*self.mat_sin,0))**2)**q)**((1/q)-1)
 
         
        
    def deriv_theta(self):
        
        """
        Method to derive coherence w.r.t theta
        """
        
        
          
        ## Direct calculation of derivative with respect to theta (u.v = u'v + uv')
       

        dPlktotal = (self.matrix.Plk[:,self.col_comb[:,0]]*self.matrix.dPlk[:,self.col_comb[:,1]] +
                     self.matrix.dPlk[:,self.col_comb[:,0]]*self.matrix.Plk[:,self.col_comb[:,1]])

         
        ## Matrix-based derivative w.r.t theta
         
        gr_temp_theta1 = (2.0*npmat.repmat(np.sum(self.ProductasLeg*self.mat_cos,0),self.matrix.m,1)*
                          (dPlktotal*self.mat_cos))
        
        gr_temp_theta2 = (2.0*npmat.repmat(np.sum(self.ProductasLeg*self.mat_sin,0),self.matrix.m,1)*
                          (dPlktotal*self.mat_sin))
        
        gr_temp_theta = gr_temp_theta1 + gr_temp_theta2
        
        ## Gradient theta
       
        self.gr_theta = self.Qnorm1*np.sum(npmat.repmat(self.Qnorm, self.matrix.m,1)*gr_temp_theta, 1)
        self.gr_theta_total = gr_temp_theta
    def deriv_phi(self):
        """
        Method to derive coherence w.r.t theta
        """
        
         
      
        ## Matrix-based derivative w.r.t phi
     
        grad_temp1 = 2.0*npmat.repmat(np.sum(self.ProductasLeg*self.mat_cos,0), self.matrix.m,1)
        grad_temp2 = self.ProductasLeg*self.mat_sin*npmat.repmat(-self.k, self.matrix.m,1)
        grad1 = grad_temp1*grad_temp2

        grad_temp3 = 2.0*npmat.repmat(np.sum(self.ProductasLeg*self.mat_sin,0), self.matrix.m,1)
        grad_temp4 = self.ProductasLeg*self.mat_cos*npmat.repmat(self.k, self.matrix.m,1)
        grad2 = grad_temp3*grad_temp4
        grad_temp = grad1 + grad2
            
        ## Gradient phi
        self.gr_phi = self.Qnorm1*np.sum(npmat.repmat(self.Qnorm, self.matrix.m,1)*grad_temp,1)
        self.gr_phi_total = grad_temp
  

      
    
class GradWigner(Gradient):
    
    
    
    """
    Class to generate derivative of product Wigner D-functions
    with respect to theta,phi, and chi.
    Since we want to optimize the coherence of the matrix (inner product of difference columns), we have
    to calculate the gradient w.r.t theta, phi and chi
    
    Attributes
    -------
    
    gr_theta, gr_phi, gr_chi: ndarray
        Gradient of the product of Wigner D-function
        with respect to angles, i.e., theta, phi, chi 
    
    
    
    """
     
        
    def comb_pos(self):
        
        """
        Method to generate combination of degree and orders
        
        """
        
        
         
        ### Combination for coherence
        comb_lkn = [self.matrix.deg_order[self.col_comb[:,0],:], self.matrix.deg_order[self.col_comb[:,1],:]]
        
        self.k = comb_lkn[0][:,1] - comb_lkn[1][:,1]
        self.n = comb_lkn[0][:,2] - comb_lkn[1][:,2]

        ## Product of combination degree and orders Wigner d-functions
        self.ProductWignerd = (self.matrix.wignerd[:,self.col_comb[:,0]]*
                               self.matrix.wignerd[:,self.col_comb[:,1]])

    
    def deriv_norm(self):
        
        """
        Method to calculate derivative of norm since we 
        approximate infinity norm for p-norm for high p
        
        """
        
        self.comb_pos()
    
        ## Allocation
        
        phi = self.matrix.angles['phi']
        chi = self.matrix.angles['chi']
         

        ## Derivative of q norm
        q = self.p

        self.mat_cos = np.cos(np.outer(phi,self.k) + np.outer(chi,self.n))
        self.mat_sin = np.sin(np.outer(phi,self.k) + np.outer(chi,self.n))

        self.Qnorm = (q/2.0)*np.sqrt(np.abs(np.sum(self.ProductWignerd*self.mat_cos,0))**2 +
                                     np.abs(np.sum(self.ProductWignerd*self.mat_sin,0))**2)**(q-2)
        self.Qnorm1 = (1/q)*np.sum(np.sqrt(np.abs(np.sum(self.ProductWignerd*self.mat_cos,0))**2 +
                                           np.abs(np.sum(self.ProductWignerd*self.mat_sin,0))**2)**q)**((1/q)-1)
 

    def deriv_theta(self):
        
        """
        Method to calculate gradient with respect 
        to theta
        
        """
        
        
        
        
        ## Direct derivative of the product with respect to theta
        dWignerd = (self.matrix.wignerd[:,self.col_comb[:,0]]*self.matrix.dwignerd[:,self.col_comb[:,1]] + 
                    self.matrix.dwignerd[:,self.col_comb[:,0]]*self.matrix.wignerd[:,self.col_comb[:,1]])
       
        
        gr_temp_theta1 = (2.0*npmat.repmat(np.sum(self.ProductWignerd*self.mat_cos,0), self.matrix.m,1)*
                          dWignerd*self.mat_cos)
        
        gr_temp_theta2 = (2.0*npmat.repmat(np.sum(self.ProductWignerd*self.mat_sin,0), self.matrix.m,1)*
                          dWignerd*self.mat_sin)
        
        gr_temp_theta = gr_temp_theta1 + gr_temp_theta2
        
        self.gr_theta = self.Qnorm1*np.sum(npmat.repmat(self.Qnorm, self.matrix.m,1)*gr_temp_theta, axis = 1)
        self.gr_theta_total = gr_temp_theta
        
    def deriv_phi(self):   
        
        """
        Method to calculate gradient with respect 
        to phi
        
        """
        
        
        
        ## Matrix-based phi
        grad_temp1_phi = 2.0*npmat.repmat(np.sum(self.ProductWignerd*self.mat_cos,0),self.matrix.m,1)
        grad_temp2_phi = (self.ProductWignerd*self.mat_sin)*npmat.repmat(-self.k, self.matrix.m,1)
        grad1_phi = (grad_temp1_phi*grad_temp2_phi)

        grad_temp3_phi = 2.0*npmat.repmat(np.sum(self.ProductWignerd*self.mat_sin,0),self.matrix.m,1)
        grad_temp4_phi = (self.ProductWignerd*self.mat_cos)*npmat.repmat(self.k,self.matrix.m,1)
        grad2_phi = (grad_temp3_phi*grad_temp4_phi)
        
        grad_temp_phi = grad1_phi + grad2_phi
        
        self.gr_phi = self.Qnorm1*np.sum((npmat.repmat(self.Qnorm, self.matrix.m,1)*grad_temp_phi), 1)
        self.gr_phi_total = grad_temp_phi
        
    def deriv_chi(self):
        
        """
        Method to calculate gradient with respect 
        to chi
        
        """
        
       
        
        ## Matrix-based chi
        grad_temp1_chi = 2.0*npmat.repmat(np.sum(self.ProductWignerd*self.mat_cos,0),self.matrix.m,1)
        grad_temp2_chi = (self.ProductWignerd*self.mat_sin)*npmat.repmat(-self.n, self.matrix.m,1)
        grad1_chi = (grad_temp1_chi*grad_temp2_chi)

        grad_temp3_chi = 2.0*npmat.repmat(np.sum(self.ProductWignerd*self.mat_sin,0),self.matrix.m,1)
        grad_temp4_chi = (self.ProductWignerd*self.mat_cos)*npmat.repmat(self.n, self.matrix.m,1)
        grad2_chi = (grad_temp3_chi*grad_temp4_chi)
        grad_temp_chi = grad1_chi + grad2_chi

        
        self.gr_chi = self.Qnorm1*np.sum((npmat.repmat(self.Qnorm,self.matrix.m,1)*grad_temp_chi), 1)
        self.gr_chi_total = grad_temp_chi
        
    def grad_total(self):
        
        """
        Method to calculate gradient total with respect
        to theta, phi, chi
        
        """
        self.deriv_norm()
        ## Calculate derivative of theta
        self.deriv_theta()
        
        ## Calculate derivative of phi
        self.deriv_phi()
        
        ## Calculate derivative of chi
        self.deriv_chi()

class GradWignerSNF(Gradient):
 
    """
    Class to calculate derivative of product Wigner D-functions for spherical near-field measurements.
    The structure is different for general case of Wigner D-functions, see in Eq ()
    We derive the product Wigner D-functions with respect to theta, phi, chi (elevation, azimuth and polarization).
    Since we want to optimize the coherence of the matrix (inner product of difference columns), we have
    to calculate the gradient w.r.t theta, phi and chi
    
    Attributes
    -------
    
    gr_theta, gr_phi, gr_chi : ndarray
        Gradient of the product of Wigner D-functions
        for spherical near-field measurements
        with respect to angles, i.e., theta, phi, chi 
    
    
    """    
        
   
       
        
    def norm_deriv(self):
 
        N = self.matrix.N//2
      
        norm_A1 = self.matrix.normA[:,0:N]
        norm_A2 = self.matrix.normA[:,N::]
        
        #### Combination    
     #   col_comb3 = self.col_comb
        self.idx_12 = np.nonzero(self.col_comb[:,1] > self.col_comb[:,0])[0]

        
        ## ProductCoh3
        ProductCoh3 = norm_A1[:,self.col_comb[:,0]]*np.conj(norm_A2[:,self.col_comb[:,1]])
        ProductCoh1 = norm_A1[:,self.col_comb[self.idx_12,0]]*np.conj(norm_A1[:,self.col_comb[self.idx_12,1]])
        ProductCoh2 = norm_A2[:,self.col_comb[self.idx_12,0]]*np.conj(norm_A2[:,self.col_comb[self.idx_12,1]])
        
       
       
        ##
        ProductCohTot = np.concatenate((np.sum(ProductCoh1,0), 
                                        np.sum(ProductCoh3,0), 
                                        np.sum(ProductCoh2,0)), axis = 0)
        ## Derivative of q norm
        q = self.p

        self.Qnorm = (q/2.0)*np.abs(ProductCohTot)**(q-2) # np.abs(np.sum(ProductCoh,0))


        self.Qnorm1 = (1/q)*np.sum(np.abs(ProductCohTot)**q)**((1/q)-1)
        
    
    def comb_orders(self):
        
        ##########################################################################
        ## For phi and chi do not need to change
        ## diferences order
    
        lk = self.matrix.deg_order
        
        comb_lk3 = [lk[self.col_comb[:,0],:],lk[self.col_comb[:,1],:]]
        
        k = comb_lk3[0][:,1] - comb_lk3[1][:,1]

        ## Combination sine and cosine 
        c1 = np.cos(np.outer(self.matrix.angles['phi'],k))
        c2 = np.cos(np.outer(self.matrix.angles['phi'],k) + 2*self.matrix.angles['chi'][:,np.newaxis])
        c3 = np.cos(np.outer(self.matrix.angles['phi'],k) - 2*self.matrix.angles['chi'][:,np.newaxis])
        s1 = np.sin(np.outer(self.matrix.angles['phi'],k))
        s2 = np.sin(np.outer(self.matrix.angles['phi'],k) + 2*self.matrix.angles['chi'][:,np.newaxis])
        s3 = np.sin(np.outer(self.matrix.angles['phi'],k) - 2*self.matrix.angles['chi'][:,np.newaxis])
        
        self.comb_sine_cos = {'c1': c1, 'c2': c2,'c3': c3,
                              's1': s1, 's2': s2,'s3': s3}

        ## Derivatives c1,c2,c3 and s1,s2,s3 (w.r.t phi)
        d_c1 = npmat.repmat(-k,self.matrix.m,1)*s1
        d_c2 = npmat.repmat(-k,self.matrix.m,1)*s2
        d_c3 = npmat.repmat(-k,self.matrix.m,1)*s3

        d_s1 = npmat.repmat(k,self.matrix.m,1)*c1
        d_s2 = npmat.repmat(k,self.matrix.m,1)*c2
        d_s3 = npmat.repmat(k,self.matrix.m,1)*c3
        
        self.deriv_sc_phi = {'d_c1': d_c1,'d_c2': d_c2,'d_c3': d_c3,
                             'd_s1': d_s1,'d_s2': d_s2,'d_s3': d_s3}
        
        ## Derivatives c1,c2,c3 and s1,s2,S3 (w.r.t chi)
        
        d_c2_chi = -s2*2
        d_c3_chi = s3*2
        d_s2_chi = c2*2
        d_s3_chi = -c3*2
        
        self.deriv_sc_chi = {'d_c2_chi': d_c2_chi, 'd_c3_chi': d_c3_chi,
                             'd_s2_chi': d_s2_chi, 'd_s3_chi': d_s3_chi}

    def all_comb_matrix(self):
        
        
        ###################################################################################
        ## Case 3 (All possible_combination of two matrices)
        #################################################################################
         
        self.comb_orders()
        
        ## Combination product Wigner small d
        d1 = self.matrix.dmm_plus1[:,self.col_comb[:,0]]*self.matrix.dmm_plus2[:,self.col_comb[:,1]]
        d2 = self.matrix.dmm_plus1[:,self.col_comb[:,0]]*self.matrix.dmm_min2[:,self.col_comb[:,1]]
        d3 = self.matrix.dmm_min1[:,self.col_comb[:,0]]*self.matrix.dmm_plus2[:,self.col_comb[:,1]]
        d4 = self.matrix.dmm_min1[:,self.col_comb[:,0]]*self.matrix.dmm_min2[:,self.col_comb[:,1]]

        ## Derivatives d1,d2,d3,d4 (w.r.t theta)
        d_d1 = (self.matrix.d_dmm_plus1[:,self.col_comb[:,0]]*self.matrix.dmm_plus2[:,self.col_comb[:,1]] + 
                self.matrix.dmm_plus1[:,self.col_comb[:,0]]*self.matrix.d_dmm_plus2[:,self.col_comb[:,1]]) 

        d_d2 = (self.matrix.d_dmm_plus1[:,self.col_comb[:,0]]*self.matrix.dmm_min2[:,self.col_comb[:,1]] +
                self.matrix.dmm_plus1[:,self.col_comb[:,0]]*self.matrix.d_dmm_min2[:,self.col_comb[:,1]])
        
        d_d3 = (self.matrix.d_dmm_min1[:,self.col_comb[:,0]]*self.matrix.dmm_plus2[:,self.col_comb[:,1]] +
                self.matrix.dmm_min1[:,self.col_comb[:,0]]*self.matrix.d_dmm_plus2[:,self.col_comb[:,1]])
        
        d_d4 = (self.matrix.d_dmm_min1[:,self.col_comb[:,0]]*self.matrix.dmm_min2[:,self.col_comb[:,1]] +
                self.matrix.dmm_min1[:,self.col_comb[:,0]]*self.matrix.d_dmm_min2[:,self.col_comb[:,1]])
              
        ## Allocation
        
        c1 = self.comb_sine_cos['c1']
        c2 = self.comb_sine_cos['c2']
        c3 = self.comb_sine_cos['c3']
             
        s1 = self.comb_sine_cos['s1']
        s2 = self.comb_sine_cos['s2']
        s3 = self.comb_sine_cos['s3']
        
        d_c1 = self.deriv_sc_phi['d_c1']
        d_c2 = self.deriv_sc_phi['d_c2']
        d_c3 = self.deriv_sc_phi['d_c3']
        
        d_s1 = self.deriv_sc_phi['d_s1']
        d_s2 = self.deriv_sc_phi['d_s2']
        d_s3 = self.deriv_sc_phi['d_s3']
        
        
        d_c2_chi =  self.deriv_sc_chi['d_c2_chi']
        d_c3_chi =  self.deriv_sc_chi['d_c3_chi']
        d_s2_chi =  self.deriv_sc_chi['d_s2_chi']
        d_s3_chi =  self.deriv_sc_chi['d_s3_chi']
        
        
                        
        
    
        ## Real
        case3_real = c1*d1 - c1*d4 - c2*d2 + c3*d3
                
        ## Deriv Real w.r.t theta
        d_case3_real_theta = c1*d_d1 - c1*d_d4 - c2*d_d2 + c3*d_d3

        ## Deriv Real w.r.t phi
        d_case3_real_phi = d_c1*d1 - d_c1*d4 - d_c2*d2 + d_c3*d3

        ## Deriv Real w.r.t chi
        d_case3_real_chi = -d_c2_chi*d2 + d_c3_chi*d3
            
        ## Imag
        case3_imag = s1*d1 - s1*d4 - s2*d2 + s3*d3


        ## Deriv Imag  w.r.t theta
        d_case3_imag_theta =  s1*d_d1 - s1*d_d4 - s2*d_d2 + s3*d_d3

        ## Deriv Imag w.r.t phi
        d_case3_imag_phi =  d_s1*d1 - d_s1*d4 - d_s2*d2 + d_s3*d3
            
        ## Deriv Imag w.r.t chi
        d_case3_imag_chi = -d_s2_chi*d2 + d_s3_chi*d3
                   
            
            
        ##Total derivation 
        self.gr_theta_case3 = case3_real*d_case3_real_theta + case3_imag*d_case3_imag_theta
        self.gr_phi_case3 = case3_real*d_case3_real_phi + case3_imag*d_case3_imag_phi
        self.gr_chi_case3 = case3_real*d_case3_real_chi + case3_imag*d_case3_imag_chi

    def comb_neg(self):
        #############################################################################
        ## Case2 (coherence inside second matrix, negative and negative)
        ############################################################################
        
        ## Use : dmm_plus2, dmm_min2. d_dmm_plus2, d_dmm_min2
        ## col_comb3, idx_12, c1,c2,c3, d_c1,d_c2,d_c3, d_c2_chi, d_c3_chi
        ## s1,s2,s3
     
        self.comb_orders()
        
        
        ## Allocation
        
        c1 = self.comb_sine_cos['c1']
        c2 = self.comb_sine_cos['c2']
        c3 = self.comb_sine_cos['c3']
             
        s1 = self.comb_sine_cos['s1']
        s2 = self.comb_sine_cos['s2']
        s3 = self.comb_sine_cos['s3']
        
        d_c1 = self.deriv_sc_phi['d_c1']
        d_c2 = self.deriv_sc_phi['d_c2']
        d_c3 = self.deriv_sc_phi['d_c3']
        
        d_s1 = self.deriv_sc_phi['d_s1']
        d_s2 = self.deriv_sc_phi['d_s2']
        d_s3 = self.deriv_sc_phi['d_s3']
        
        
        d_c2_chi =  self.deriv_sc_chi['d_c2_chi']
        d_c3_chi =  self.deriv_sc_chi['d_c3_chi']
        d_s2_chi =  self.deriv_sc_chi['d_s2_chi']
        d_s3_chi =  self.deriv_sc_chi['d_s3_chi']
        
        
        ## Combination product Wigner small d
        d1 = self.matrix.dmm_plus2[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_plus2[:,self.col_comb[self.idx_12,1]]
        d2 = self.matrix.dmm_plus2[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_min2[:,self.col_comb[self.idx_12,1]]
        d3 = self.matrix.dmm_min2[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_plus2[:,self.col_comb[self.idx_12,1]]
        d4 = self.matrix.dmm_min2[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_min2[:,self.col_comb[self.idx_12,1]]

        ## Derivatives d1,d2,d3,d4 (w.r.t theta)
        d_d1 = (self.matrix.d_dmm_plus2[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_plus2[:,self.col_comb[self.idx_12,1]] + 
                self.matrix.dmm_plus2[:,self.col_comb[self.idx_12,0]]*self.matrix.d_dmm_plus2[:, self.col_comb[self.idx_12,1]]) 
        d_d2 = (self.matrix.d_dmm_plus2[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_min2[:,self.col_comb[self.idx_12,1]] +
                self.matrix.dmm_plus2[:,self.col_comb[self.idx_12,0]]*self.matrix.d_dmm_min2[:,self.col_comb[self.idx_12,1]])
        d_d3 = (self.matrix.d_dmm_min2[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_plus2[:,self.col_comb[self.idx_12,1]] +
                self.matrix.dmm_min2[:,self.col_comb[self.idx_12,0]]*self.matrix.d_dmm_plus2[:,self.col_comb[self.idx_12,1]])
        d_d4 = (self.matrix.d_dmm_min2[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_min2[:,self.col_comb[self.idx_12,1]] +
                self.matrix.dmm_min2[:,self.col_comb[self.idx_12,0]]*self.matrix.d_dmm_min2[:,self.col_comb[self.idx_12,1]])
        
        ## Real
        case2_real = (c1[:,self.idx_12]*d1  + c1[:,self.idx_12]*d4 -
                      c2[:,self.idx_12]*d2  - c3[:,self.idx_12]*d3)

        ## Deriv Real w.r.t theta
        d_case2_real_theta = (c1[:,self.idx_12]*d_d1 + 
                              c1[:,self.idx_12]*d_d4 - 
                              c2[:,self.idx_12]*d_d2 - 
                              c3[:,self.idx_12]*d_d3)

        ## Deriv Real w.r.t phi
        d_case2_real_phi = (d_c1[:,self.idx_12]*d1 + 
                            d_c1[:,self.idx_12]*d4 - 
                            d_c2[:,self.idx_12]*d2 - 
                            d_c3[:,self.idx_12]*d3)
            
            
        ## Deriv Real w.r.t chi
        d_case2_real_chi = (- d_c2_chi[:,self.idx_12]*d2 - 
                              d_c3_chi[:,self.idx_12]*d3)
            
            

        ## Imag
        case2_imag = (s1[:,self.idx_12]*d1 + 
                      s1[:,self.idx_12]*d4 - 
                      s2[:,self.idx_12]*d2 - 
                      s3[:,self.idx_12]*d3)


        ## Deriv Imag  w.r.t theta
        d_case2_imag_theta =  (s1[:,self.idx_12]*d_d1 +
                               s1[:,self.idx_12]*d_d4 - 
                               s2[:,self.idx_12]*d_d2 - 
                               s3[:,self.idx_12]*d_d3)

        ## Deriv Imag w.r.t phi
        d_case2_imag_phi =  (d_s1[:,self.idx_12]*d1 +
                             d_s1[:,self.idx_12]*d4 - 
                             d_s2[:,self.idx_12]*d2 - 
                             d_s3[:,self.idx_12]*d3)
            
        ## Deriv Imag w.r.t chi
        d_case2_imag_chi = (-d_s2_chi[:,self.idx_12]*d2 - 
                             d_s3_chi[:,self.idx_12]*d3)


        ##Total derivation
        self.gr_theta_case2 = case2_real*d_case2_real_theta + case2_imag*d_case2_imag_theta
        self.gr_phi_case2   = case2_real*d_case2_real_phi + case2_imag*d_case2_imag_phi
        self.gr_chi_case2   = case2_real*d_case2_real_chi + case2_imag*d_case2_imag_chi 
        
    def comb_pos(self):

        ###############################################################################
        ## Case 1 (Coherence between basis 1, positive and positive)
        ###############################################################################

        self.comb_orders()
        ## Use : dmm_plus1, dmm_min1. d_dmm_plus1, d_dmm_min1
        ## col_comb3, idx_12, c1,c2,c3, d_c1,d_c2,d_c3, d_c2_chi, d_c3_chi
        ## s1,s2,s3
        
        ## Allocation
        
        c1 = self.comb_sine_cos['c1']
        c2 = self.comb_sine_cos['c2']
        c3 = self.comb_sine_cos['c3']
             
        s1 = self.comb_sine_cos['s1']
        s2 = self.comb_sine_cos['s2']
        s3 = self.comb_sine_cos['s3']
        
        d_c1 = self.deriv_sc_phi['d_c1']
        d_c2 = self.deriv_sc_phi['d_c2']
        d_c3 = self.deriv_sc_phi['d_c3']
        
        d_s1 = self.deriv_sc_phi['d_s1']
        d_s2 = self.deriv_sc_phi['d_s2']
        d_s3 = self.deriv_sc_phi['d_s3']
        
        
        d_c2_chi =  self.deriv_sc_chi['d_c2_chi']
        d_c3_chi =  self.deriv_sc_chi['d_c3_chi']
        d_s2_chi =  self.deriv_sc_chi['d_s2_chi']
        d_s3_chi =  self.deriv_sc_chi['d_s3_chi']
     
        
        ## Combination product Wigner small d
        d1 = self.matrix.dmm_plus1[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_plus1[:,self.col_comb[self.idx_12,1]]
        d2 = self.matrix.dmm_plus1[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_min1[:,self.col_comb[self.idx_12,1]]
        d3 = self.matrix.dmm_min1[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_plus1[:,self.col_comb[self.idx_12,1]]
        d4 = self.matrix.dmm_min1[:,self.col_comb[self.idx_12,0]]*self.matrix.dmm_min1[:,self.col_comb[self.idx_12,1]]

        ## Derivatives d1,d2,d3,d4 (w.r.t theta)
        d_d1 = (self.matrix.d_dmm_plus1[:,self.col_comb[self.idx_12,0]]*
                self.matrix.dmm_plus1[:,self.col_comb[self.idx_12,1]] + 
                self.matrix.dmm_plus1[:,self.col_comb[self.idx_12,0]]* 
                self.matrix.d_dmm_plus1[:, self.col_comb[self.idx_12,1]])
        
        d_d2 = (self.matrix.d_dmm_plus1[:,self.col_comb[self.idx_12,0]]*
                self.matrix.dmm_min1[:,self.col_comb[self.idx_12,1]] +
                self.matrix.dmm_plus1[:,self.col_comb[self.idx_12,0]]*
                self.matrix.d_dmm_min1[:,self.col_comb[self.idx_12,1]])
        
        d_d3 = (self.matrix.d_dmm_min1[:,self.col_comb[self.idx_12,0]]*
                self.matrix.dmm_plus1[:,self.col_comb[self.idx_12,1]] +
                self.matrix.dmm_min1[:,self.col_comb[self.idx_12,0]]*
                self.matrix.d_dmm_plus1[:,self.col_comb[self.idx_12,1]])
        
        d_d4 = (self.matrix.d_dmm_min1[:,self.col_comb[self.idx_12,0]]*
                self.matrix.dmm_min1[:,self.col_comb[self.idx_12,1]] +
                self.matrix.dmm_min1[:,self.col_comb[self.idx_12,0]]*
                self.matrix.d_dmm_min1[:,self.col_comb[self.idx_12,1]])
        
        ## Real
        case1_real = (c1[:,self.idx_12]*d1 + c1[:,self.idx_12]*d4 +
                      c2[:,self.idx_12]*d2 + c3[:,self.idx_12]*d3)

        ## Deriv Real w.r.t theta
        d_case1_real_theta = (c1[:,self.idx_12]*d_d1 +
                              c1[:,self.idx_12]*d_d4 +
                              c2[:,self.idx_12]*d_d2 +
                              c3[:,self.idx_12]*d_d3)

        ## Deriv Real w.r.t phi
        d_case1_real_phi = (d_c1[:,self.idx_12]*d1 +
                            d_c1[:,self.idx_12]*d4 +
                            d_c2[:,self.idx_12]*d2 +
                            d_c3[:,self.idx_12]*d3)
            
        ## Deriv Real w.r.t chi
        d_case1_real_chi =  (d_c2_chi[:,self.idx_12]*d2 +
                             d_c3_chi[:,self.idx_12]*d3)

            
        ## Imag
        case1_imag = (s1[:,self.idx_12]*d1 +
                      s1[:,self.idx_12]*d4 +
                      s2[:,self.idx_12]*d2 +
                      s3[:,self.idx_12]*d3)


        ## Deriv Imag  w.r.t theta
        d_case1_imag_theta =  (s1[:,self.idx_12]*d_d1 +
                               s1[:,self.idx_12]*d_d4 +
                               s2[:,self.idx_12]*d_d2 +
                               s3[:,self.idx_12]*d_d3)

        ## Deriv Imag w.r.t phi
        d_case1_imag_phi =  (d_s1[:,self.idx_12]*d1 +
                             d_s1[:,self.idx_12]*d4 +
                             d_s2[:,self.idx_12]*d2 +
                             d_s3[:,self.idx_12]*d3)
            
        ## Deriv Imag w.r.t chi
        d_case1_imag_chi =  (d_s2_chi[:,self.idx_12]*d2 +
                             d_s3_chi[:,self.idx_12]*d3)
            
        ##Total derivation(DONT FORGET NORMALIZATION)
        self.gr_theta_case1 = case1_real*d_case1_real_theta + case1_imag*d_case1_imag_theta
        self.gr_phi_case1   = case1_real*d_case1_real_phi + case1_imag*d_case1_imag_phi
        self.gr_chi_case1   = case1_real*d_case1_real_chi + case1_imag*d_case1_imag_chi

    def grad_all(self):
        self.all_comb_matrix()
        self.norm_deriv()
        self.comb_pos()
        self.comb_neg()
        
        ## Concatenate total
        self.gr_theta_total = (np.concatenate((self.gr_theta_case1,
                                               self.gr_theta_case3,
                                               self.gr_theta_case2), axis = 1))
              
        self.gr_phi_total = (np.concatenate((self.gr_phi_case1, 
                                             self.gr_phi_case3, 
                                             self.gr_phi_case2), axis = 1))
 
        self.gr_chi_total = (np.concatenate((self.gr_chi_case1,
                                             self.gr_chi_case3,
                                             self.gr_chi_case2), axis = 1))
           
    def grad_total(self):
       
    
        self.grad_all()                                     
        
        ## Qnorm, Qnorm1, gr_theta_total, gr_phi_total
        self.gr_theta = self.Qnorm1*np.sum(npmat.repmat(self.Qnorm,self.matrix.m,1)*self.gr_theta_total, 1)
        self.gr_phi   = self.Qnorm1*np.sum(npmat.repmat(self.Qnorm,self.matrix.m,1)*self.gr_phi_total, 1)
        self.gr_chi   = self.Qnorm1*np.sum(npmat.repmat(self.Qnorm,self.matrix.m,1)*self.gr_chi_total, 1)
          
   

