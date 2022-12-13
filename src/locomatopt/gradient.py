import numpy as np 


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
       

class GradSH(Gradient):
    
    """
    Class to generate gradient of product combination columns of 
    spherical harmonics matrix. 
    Since we want to optimize the coherence of the matrix (inner product of difference columns), 
    we have to calculate the gradient w.r.t theta, phi
    
    
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
        k, ProductasLeg = self.comb_pos()
        Qnorm, Qnorm1, mat_cos, mat_sin = self.deriv_norm(k, ProductasLeg)

        # Calculate derivative of theta
        gr_theta_pnorm, gr_theta = self.deriv_theta(Qnorm1, Qnorm, ProductasLeg,
                                                    mat_cos, mat_sin)
        
        # Calculate derivative of phi
        gr_phi_pnorm, gr_phi = self.deriv_phi(k, Qnorm, Qnorm1, ProductasLeg,
                                              mat_cos, mat_sin)
        
        return {'theta': gr_theta_pnorm,
                'phi': gr_phi_pnorm,
                'theta_func': gr_theta,
                'phi_func': gr_phi}

    def comb_pos(self):
        """
        Method to generate combination of degree and orders
        
        """
         
        # Combination of degree and orders
        comb_lk = [self.matrix.deg_order[self.col_comb[:, 0], :],
                   self.matrix.deg_order[self.col_comb[:, 1], :]]
        
        # Product of combination of degree and order associated Legendre
        ProductasLeg = (self.matrix.Plk[:, self.col_comb[:, 0]] *
                        self.matrix.Plk[:, self.col_comb[:, 1]])
        
        # Differences order
        k = comb_lk[0][:, 1] - comb_lk[1][:, 1]
        return k, ProductasLeg

    def deriv_norm(self, k, ProductasLeg):
        
        """
        Method to calculate derivative of norm since we 
        approximate infinity norm for p-norm for high p
        """
        
        # Allocation
       
        phi = self.matrix.angles['phi']
        
        # q-norm, q = 8
        q = self.p
         
        # Derivative of norm
        
        mat_cos = np.cos(np.outer(phi, k))
        mat_sin = np.sin(np.outer(phi, k))
        
        abs_product = np.sqrt(np.abs(np.sum(ProductasLeg*mat_cos, 0))**2 +
                              np.abs(np.sum(ProductasLeg*mat_sin, 0))**2)

        Qnorm = (q/2.0)*abs_product**(q-2)
    
        Qnorm1 = (1/q)*(np.sum(abs_product**q)**((1/q)-1))
        
        return Qnorm, Qnorm1, mat_cos, mat_sin

    def deriv_theta(self, Qnorm1, Qnorm, ProductasLeg,
                    mat_cos, mat_sin):
        
        """
        Method to derive coherence w.r.t theta
        """
        
        # Direct calculation of derivative with respect to theta (u.v = u'v + uv')
        dPlktotal = (self.matrix.Plk[:, self.col_comb[:, 0]] *
                     self.matrix.dPlk[:, self.col_comb[:, 1]] +
                     self.matrix.dPlk[:, self.col_comb[:, 0]] * 
                     self.matrix.Plk[:, self.col_comb[:, 1]])

        # Matrix-based derivative w.r.t theta 

        gr_theta = (2.0*np.sum(ProductasLeg*mat_cos, 0)[np.newaxis, :] *
                    dPlktotal*mat_cos +
                    2.0*np.sum(ProductasLeg*mat_sin, 0)[np.newaxis, :] *
                    dPlktotal*mat_sin)
        
        # Gradient theta
        
        gr_theta_pnorm = Qnorm1*np.sum(Qnorm[np.newaxis, :] *
                                       gr_theta, 1)
        
        return gr_theta_pnorm, gr_theta

    def deriv_phi(self, k, Qnorm, Qnorm1, ProductasLeg, mat_cos, mat_sin):
        """
        Method to derive coherence w.r.t theta
        """
        
        # Matrix-based derivative w.r.t phi 
        
        gr_phi = ((2.0*np.sum(ProductasLeg*mat_cos, 0)[np.newaxis, :] *
                   ProductasLeg*mat_sin*(-k[np.newaxis, :])) + 
                  (2.0*np.sum(ProductasLeg*mat_sin, 0)[np.newaxis, :] * 
                   ProductasLeg*mat_cos*(k[np.newaxis, :])))
                  
        # Gradient phi
        gr_phi_pnorm = Qnorm1*np.sum(Qnorm[np.newaxis, :]*gr_phi, 1)
  
        return gr_phi_pnorm, gr_phi
 

class GradWigner(Gradient):
    
    """
    Class to generate derivative of product Wigner D-functions
    with respect to theta,phi, and chi.
    Since we want to optimize the coherence of the matrix (inner product of difference columns),
    we have to calculate the gradient w.r.t theta, phi and chi
    
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
        
        # Combination for coherence
        comb_lkn = [self.matrix.deg_order[self.col_comb[:, 0], :],
                    self.matrix.deg_order[self.col_comb[:, 1], :]]
        
        k = comb_lkn[0][:, 1] - comb_lkn[1][:, 1]
        n = comb_lkn[0][:, 2] - comb_lkn[1][:, 2]

        # Product of combination degree and orders Wigner d-functions
        ProductWignerd = (self.matrix.wignerd[:, self.col_comb[:, 0]] *
                          self.matrix.wignerd[:, self.col_comb[:, 1]])
        
        return k, n, ProductWignerd

    def deriv_norm(self, k, n, ProductWignerd):
        
        """
        Method to calculate derivative of norm since we 
        approximate infinity norm for p-norm for high p
        
        """
         
        # Allocation
        phi = self.matrix.angles['phi']
        chi = self.matrix.angles['chi']
        
        # Derivative of q norm
        q = self.p

        mat_cos = np.cos(np.outer(phi, k) + np.outer(chi, n))
        mat_sin = np.sin(np.outer(phi, k) + np.outer(chi, n))

        abs_product = np.sqrt(np.abs(np.sum(ProductWignerd*mat_cos, 0))**2 +
                              np.abs(np.sum(ProductWignerd*mat_sin, 0))**2)

        Qnorm = (q/2.0)*abs_product**(q-2)
        Qnorm1 = (1/q)*(np.sum(abs_product**q)**((1/q)-1))

        return Qnorm, Qnorm1, mat_cos, mat_sin

    def deriv_theta(self, Qnorm1, Qnorm, ProductWignerd,
                    mat_cos, mat_sin):
        
        """
        Method to calculate gradient with respect 
        to theta
        
        """
    
        # Direct derivative of the product with respect to theta
        dWignerd = (self.matrix.wignerd[:, self.col_comb[:, 0]] *
                    self.matrix.dwignerd[:, self.col_comb[:, 1]] +
                    self.matrix.dwignerd[:, self.col_comb[:, 0]] *
                    self.matrix.wignerd[:, self.col_comb[:, 1]])
        
        gr_theta = (2.0*np.sum(ProductWignerd*mat_cos, 0)[np.newaxis, :] *
                    dWignerd*mat_cos + 
                    2.0*np.sum(ProductWignerd*mat_sin, 0)[np.newaxis, :] *
                    dWignerd*mat_sin)

        gr_theta_pnorm = Qnorm1*np.sum(Qnorm[np.newaxis, :] *
                                       gr_theta, 1)

        return gr_theta_pnorm, gr_theta

    def deriv_phi(self, k, Qnorm, Qnorm1, ProductWignerd, mat_cos, mat_sin):   
        
        """
        Method to calculate gradient with respect 
        to phi
        
        """
        
        # Matrix-based phi
        
        gr_phi = (2.0*np.sum(ProductWignerd*mat_cos, 0)[np.newaxis, :] * 
                  ProductWignerd*mat_sin*(-k[np.newaxis, :]) + 
                  2.0*np.sum(ProductWignerd*mat_sin, 0)[np.newaxis, :] *
                  ProductWignerd*mat_cos*(k[np.newaxis, :]))
        
        gr_phi_pnorm = Qnorm1*np.sum(Qnorm[np.newaxis, :]*gr_phi, 1)
        
        return gr_phi_pnorm, gr_phi
        
    def deriv_chi(self, n, Qnorm, Qnorm1, ProductWignerd, mat_cos, mat_sin):
        
        """
        Method to calculate gradient with respect 
        to chi
        
        """
       
        # Matrix-based chi
        gr_chi = (2.0*np.sum(ProductWignerd*mat_cos, 0)[np.newaxis, :] *
                  ProductWignerd*mat_sin*(-n[np.newaxis, :]) +
                  2.0*np.sum(ProductWignerd*mat_sin, 0)[np.newaxis, :] *     
                  ProductWignerd*mat_cos*(n[np.newaxis, :]))
       
        gr_chi_pnorm = Qnorm1*np.sum(Qnorm[np.newaxis, :]*gr_chi, 1)
        
        return gr_chi_pnorm, gr_chi
        
    def grad_total(self):
        
        """
        Method to calculate gradient total with respect
        to theta, phi, chi
        
        """

       
        k, n, ProductWignerd = self.comb_pos()

        Qnorm, Qnorm1, mat_cos, mat_sin = self.deriv_norm(k, n, ProductWignerd)
        # Calculate derivative of theta
        gr_theta_pnorm, gr_theta = self.deriv_theta(Qnorm1, Qnorm, ProductWignerd,
                                                    mat_cos, mat_sin)
    
        # Calculate derivative of phi
        gr_phi_pnorm, gr_phi = self.deriv_phi(k, Qnorm, Qnorm1, ProductWignerd, 
                                              mat_cos, mat_sin)
        
        # Calculate derivative of chi
        gr_chi_pnorm, gr_chi = self.deriv_chi(n, Qnorm, Qnorm1, ProductWignerd, 
                                              mat_cos, mat_sin)

        return {'theta': gr_theta_pnorm,
                'phi': gr_phi_pnorm,
                'chi': gr_chi_pnorm,
                'theta_func': gr_theta,
                'phi_func': gr_phi,
                'chi_func': gr_chi}

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

        Qnorm = (q/2.0)*np.abs(ProductCohTot)**(q-2) # np.abs(np.sum(ProductCoh,0))


        Qnorm1 = (1/q)*np.sum(np.abs(ProductCohTot)**q)**((1/q)-1)
        
        return Qnorm, Qnorm1
    
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
        
        comb_sine_cos = {'c1': c1, 'c2': c2,'c3': c3,
                         's1': s1, 's2': s2,'s3': s3}

        ## Derivatives c1,c2,c3 and s1,s2,s3 (w.r.t phi)
        d_c1 = -k[np.newaxis, :]*s1
        d_c2 = -k[np.newaxis, :]*s2
        d_c3 = -k[np.newaxis, :]*s3

        d_s1 = k[np.newaxis, :]*c1
        d_s2 = k[np.newaxis, :]*c2
        d_s3 = k[np.newaxis, :]*c3
        
        deriv_sc_phi = {'d_c1': d_c1,'d_c2': d_c2,'d_c3': d_c3,
                        'd_s1': d_s1,'d_s2': d_s2,'d_s3': d_s3}
        
        ## Derivatives c1,c2,c3 and s1,s2,S3 (w.r.t chi)
        
        d_c2_chi = -s2*2
        d_c3_chi = s3*2
        d_s2_chi = c2*2
        d_s3_chi = -c3*2
        
        deriv_sc_chi = {'d_c2_chi': d_c2_chi, 'd_c3_chi': d_c3_chi,
                        'd_s2_chi': d_s2_chi, 'd_s3_chi': d_s3_chi}
        
        del c1,c2,c3,s1,s2,s3, d_c1, d_c2, d_c3, d_s1, d_s2, d_s3, d_c2_chi,d_c3_chi, d_s2_chi, d_s3_chi
        return deriv_sc_phi, deriv_sc_chi, comb_sine_cos

    def all_comb_matrix(self, comb_sine_cos, deriv_sc_chi, deriv_sc_phi):
        
        
        ###################################################################################
        ## Case 3 (All possible_combination of two matrices)
        #################################################################################
         
        #self.comb_orders()
        
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
        
        c1 = comb_sine_cos['c1']
        c2 = comb_sine_cos['c2']
        c3 = comb_sine_cos['c3']
             
        s1 = comb_sine_cos['s1']
        s2 = comb_sine_cos['s2']
        s3 = comb_sine_cos['s3']
        
        d_c1 = deriv_sc_phi['d_c1']
        d_c2 = deriv_sc_phi['d_c2']
        d_c3 = deriv_sc_phi['d_c3']
        
        d_s1 = deriv_sc_phi['d_s1']
        d_s2 = deriv_sc_phi['d_s2']
        d_s3 = deriv_sc_phi['d_s3']
        
        
        d_c2_chi =  deriv_sc_chi['d_c2_chi']
        d_c3_chi =  deriv_sc_chi['d_c3_chi']
        d_s2_chi =  deriv_sc_chi['d_s2_chi']
        d_s3_chi =  deriv_sc_chi['d_s3_chi']
        
        
                        
        
    
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
        gr_theta_case3 = case3_real*d_case3_real_theta + case3_imag*d_case3_imag_theta
        gr_phi_case3 = case3_real*d_case3_real_phi + case3_imag*d_case3_imag_phi
        gr_chi_case3 = case3_real*d_case3_real_chi + case3_imag*d_case3_imag_chi

        del d1, d2,d3, d4, d_d1, d_d2, d_d3, d_d4, c1,c2,c3, s1,s2,s3, d_c1, d_c2,  d_c3
        del d_s1, d_s2, d_s3 , d_c2_chi, d_c3_chi, d_s2_chi, d_s3_chi
        del case3_real, d_case3_real_theta, d_case3_real_phi, d_case3_real_chi 
        del case3_imag, d_case3_imag_theta, d_case3_imag_phi, d_case3_imag_chi 
        return gr_theta_case3, gr_phi_case3, gr_chi_case3

    def comb_neg(self, comb_sine_cos, deriv_sc_phi, deriv_sc_chi):
        #############################################################################
        ## Case2 (coherence inside second matrix, negative and negative)
        ############################################################################
        
        ## Use : dmm_plus2, dmm_min2. d_dmm_plus2, d_dmm_min2
        ## col_comb3, idx_12, c1,c2,c3, d_c1,d_c2,d_c3, d_c2_chi, d_c3_chi
        ## s1,s2,s3
     
        #self.comb_orders()
        
        
        ## Allocation
        
        c1 = comb_sine_cos['c1']
        c2 = comb_sine_cos['c2']
        c3 = comb_sine_cos['c3']
             
        s1 = comb_sine_cos['s1']
        s2 = comb_sine_cos['s2']
        s3 = comb_sine_cos['s3']
        
        d_c1 = deriv_sc_phi['d_c1']
        d_c2 = deriv_sc_phi['d_c2']
        d_c3 = deriv_sc_phi['d_c3']
        
        d_s1 = deriv_sc_phi['d_s1']
        d_s2 = deriv_sc_phi['d_s2']
        d_s3 = deriv_sc_phi['d_s3']
        
        
        d_c2_chi = deriv_sc_chi['d_c2_chi']
        d_c3_chi = deriv_sc_chi['d_c3_chi']
        d_s2_chi = deriv_sc_chi['d_s2_chi']
        d_s3_chi = deriv_sc_chi['d_s3_chi']
        
        
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
        gr_theta_case2 = case2_real*d_case2_real_theta + case2_imag*d_case2_imag_theta
        gr_phi_case2   = case2_real*d_case2_real_phi + case2_imag*d_case2_imag_phi
        gr_chi_case2   = case2_real*d_case2_real_chi + case2_imag*d_case2_imag_chi 

        del d1, d2,d3, d4, d_d1, d_d2, d_d3, d_d4, c1,c2,c3, s1,s2,s3, d_c1, d_c2,  d_c3
        del d_s1, d_s2, d_s3 , d_c2_chi, d_c3_chi, d_s2_chi, d_s3_chi
        del case2_real, d_case2_real_theta, d_case2_real_phi, d_case2_real_chi 
        del case2_imag, d_case2_imag_theta, d_case2_imag_phi, d_case2_imag_chi 
        return gr_theta_case2, gr_phi_case2, gr_chi_case2

    def comb_pos(self, comb_sine_cos, deriv_sc_phi, deriv_sc_chi):

        ##############################################################################
        # Case 1 (Coherence between basis 1, positive and positive)
        ###############################################################################

        #self.comb_orders()
        ## Use : dmm_plus1, dmm_min1. d_dmm_plus1, d_dmm_min1
        ## col_comb3, idx_12, c1,c2,c3, d_c1,d_c2,d_c3, d_c2_chi, d_c3_chi
        ## s1,s2,s3
        
        ## Allocation
        
        c1 = comb_sine_cos['c1']
        c2 = comb_sine_cos['c2']
        c3 = comb_sine_cos['c3']
             
        s1 = comb_sine_cos['s1']
        s2 = comb_sine_cos['s2']
        s3 = comb_sine_cos['s3']
        
        d_c1 = deriv_sc_phi['d_c1']
        d_c2 = deriv_sc_phi['d_c2']
        d_c3 = deriv_sc_phi['d_c3']
        
        d_s1 = deriv_sc_phi['d_s1']
        d_s2 = deriv_sc_phi['d_s2']
        d_s3 = deriv_sc_phi['d_s3']
        
        
        d_c2_chi =  deriv_sc_chi['d_c2_chi']
        d_c3_chi =  deriv_sc_chi['d_c3_chi']
        d_s2_chi =  deriv_sc_chi['d_s2_chi']
        d_s3_chi =  deriv_sc_chi['d_s3_chi']
     
        
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
        gr_theta_case1 = case1_real*d_case1_real_theta + case1_imag*d_case1_imag_theta
        gr_phi_case1   = case1_real*d_case1_real_phi + case1_imag*d_case1_imag_phi
        gr_chi_case1   = case1_real*d_case1_real_chi + case1_imag*d_case1_imag_chi

        del d1, d2,d3, d4, d_d1, d_d2, d_d3, d_d4, c1,c2,c3, s1,s2,s3, d_c1, d_c2,  d_c3
        del d_s1, d_s2, d_s3 , d_c2_chi, d_c3_chi, d_s2_chi, d_s3_chi
        del case1_real, d_case1_real_theta, d_case1_real_phi, d_case1_real_chi 
        del case1_imag, d_case1_imag_theta, d_case1_imag_phi, d_case1_imag_chi 
        return gr_theta_case1, gr_phi_case1, gr_chi_case1


    def grad_all(self):

        deriv_sc_phi, deriv_sc_chi, comb_sine_cos = self.comb_orders()

        gr_theta_case3, gr_phi_case3, gr_chi_case3 = self.all_comb_matrix(comb_sine_cos, deriv_sc_chi, deriv_sc_phi)
        
        gr_theta_case2, gr_phi_case2, gr_chi_case2 = self.comb_neg(comb_sine_cos, deriv_sc_phi, deriv_sc_chi)
        
        gr_theta_case1, gr_phi_case1, gr_chi_case1 = self.comb_pos(comb_sine_cos, deriv_sc_phi, deriv_sc_chi)
        
         
        ## Concatenate total
        gr_theta = (np.concatenate((gr_theta_case1,
                                          gr_theta_case3,
                                          gr_theta_case2), axis = 1))
              
        gr_phi = (np.concatenate((gr_phi_case1, 
                                        gr_phi_case3, 
                                        gr_phi_case2), axis = 1))
 
        gr_chi = (np.concatenate((gr_chi_case1,
                                        gr_chi_case3,
                                        gr_chi_case2), axis = 1))

        del deriv_sc_phi, deriv_sc_chi, comb_sine_cos, gr_theta_case3, gr_phi_case3, gr_chi_case3 
        del gr_theta_case2, gr_phi_case2, gr_chi_case2, gr_theta_case1, gr_phi_case1, gr_chi_case1 
        
        return gr_theta, gr_phi, gr_chi
           
    def grad_total(self):
       
        Qnorm, Qnorm1 = self.norm_deriv()                                
        
        gr_theta, gr_phi, gr_chi = self.grad_all()
        
        ## Qnorm, Qnorm1, gr_theta_total, gr_phi_total
        gr_theta_pnorm = Qnorm1*np.sum(Qnorm[np.newaxis, :]*gr_theta, 1)
        gr_phi_pnorm   = Qnorm1*np.sum(Qnorm[np.newaxis, :]*gr_phi, 1)
        gr_chi_pnorm   = Qnorm1*np.sum(Qnorm[np.newaxis, :]*gr_chi, 1)


        del Qnorm, Qnorm1
        return {'theta': gr_theta_pnorm,
                'phi': gr_phi_pnorm,
                'chi': gr_chi_pnorm,
                'theta_func': gr_theta,
                'phi_func': gr_phi,
                'chi_func': gr_chi}
          
   

