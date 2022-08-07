import math
import numpy as np
from scipy.special import sph_harm as SH
from scipy.special import eval_jacobi as Plkn
from numpy import linalg as LA
np.seterr(divide='ignore', invalid='ignore')


class Matrix:
    """ 
    Base class to construct matrix from spherical harmonics and Wigner D-functions.
      
    Parameters
    ----------
    B : int
        Bandlimited for degree and order
           
    angles: ndarray
        Sampling points to generate matrix
        
  
    """
    def __init__(self, B, angles, case, complex_dtype=np.complex128, float_dtype=np.float64):
        self.B = B
        self.angles = angles
        self.m = len(angles['theta'])
        self.case = case
        self.complex_dtype = complex_dtype
        self.float_dtype = float_dtype
        self.gen_matrix()

    def degree_orders(self):
        pass
    
    def gen_matrix(self):
        pass
     

class MatrixSH(Matrix):
    
    """ 
    Class to construct matrix from spherical harmonics
      
    Attributes
    -------
    A : ndarray
        Matrix from spherical harmonics 
    normA : ndarray
        Normalized matrix from spherical harmonics 
        w.r.t l2-norm
    
    dPlk : ndarray
        Matrix derivative of associated Legendre polynomials
    
    Plk : ndarray
        Matrix associated Legendre polynomials
    
    m : int
        row dimension of the matrix (size samples)
    N : int
        col dimension of the matrix (combination of degree and orders)
    deg_order : ndarray
        combination of degre and order    
   
   
    """        
            
    def degree_orders(self):
        '''
        Generating combination of degree and orders for spherical
        harmonics
        '''
        self.N = self.B**2
        lk = np.zeros((self.N, 2), dtype=self.float_dtype)
        idx_beg = 0
        for l_deg in range(self.B):
            k = range(-l_deg, l_deg+1)
            idx = len(k)
            idx_end = idx_beg + idx-1
            lk[idx_beg:idx_end+1, 1] = k
            lk[idx_beg:idx_end+1, 0] = np.full((1, idx), l_deg)
            idx_beg = idx_beg + idx
        self.deg_order = lk
    
    def asleg(self, ii, theta, phi_0):
        
        """
        Method to generate associated Legendre polynomials
        matrix. The method calls spherical harmonics function
        The associated Legendre polynomials is a real function
        and can be generated from spherical harmonics phi = 0
        """
        
        # Get Associated Legendre (get real since we have phi_0 = 0)
        # note that absolute value spherical harmonics = absolute value associated Legendre
       
        return (np.real(SH(self.deg_order[ii, 1], self.deg_order[ii, 0], phi_0, theta)) /
                LA.norm(np.real(SH(self.deg_order[ii, 1], self.deg_order[ii, 0], phi_0, theta))))
                
    def deriv_asleg(self, ii, theta, phi_0, Plk):
        """
        Method to generate derivative of associated Legendre
        polynomials
        
        """
        
        # Calculate derivative w.r.t theta if choose all
        # Spherical harmonics order > degree, 
        # just assign arbitrary since will be zero multiply with the coefficients
               
        if self.deg_order[ii, 1] + 1 > self.deg_order[ii, 0]:
            Plk_lastterm = np.ones(self.m)
        else:
            # (get real since we have phi_0 = 0)
            Plk_lastterm = (np.real(np.exp(-1j*phi_0)*SH(self.deg_order[ii, 1] + 1,
                                                         self.deg_order[ii, 0], phi_0, theta)) /
                            LA.norm(SH(self.deg_order[ii, 1] + 1,
                                    self.deg_order[ii, 0], phi_0, theta)))
                        
        # Derivative of spherical harmonics w.r.t theta, or derivative of associated Legendre
        # functions
        Plk_deriv = ((self.deg_order[ii, 1]/np.tan(theta))*Plk +
                     np.sqrt((self.deg_order[ii, 0] - self.deg_order[ii, 1]) *
                             (self.deg_order[ii, 0] + self.deg_order[ii, 1] + 1)) *
                     Plk_lastterm)     
        
        return Plk_deriv
    
    def gen_matrix(self):
        
        """
           Method to generate Spherical Harmonics Matrix and their derivative
           with respect to theta (Derivative of associated Legendre polynomials)
           d/dtheta ) = k/tan(theta) Ylk(theta,phi) + sqrt((l-k)(l+k+1))*Yl(k+1)(theta,phi)
           The spherical harmonics when k+1 > l will be zero because of multiplication with 
           sqrt((l-k)(l+k+1)
        
        """
        
        # Generate degree and order
        self.degree_orders()
        
        #  Generate matrix
        A = np.zeros((self.m, self.N), dtype=self.complex_dtype)
        normA = np.zeros_like(A)
        dPlk = np.zeros((self.m, self.N), dtype=self.float_dtype)
        Plk = np.copy(dPlk)
        
        theta = self.angles['theta']
        phi = self.angles['phi']
        
        # Asign phi = 0 to get associated Legendre
        phi_0 = np.zeros(self.m, dtype=self.float_dtype)
       
        for ii in range(self.N):
            # Generate spherical harmonics and their (column) normalization w.r.t sampling points
            A[:, ii] = SH(self.deg_order[ii, 1], self.deg_order[ii, 0], phi, theta)
             
            # Normalize w.r.t l2 norm
            normA[:, ii] = A[:, ii]/LA.norm(A[:, ii])
            
            # Associated Legendre Polynomials
            Plk[:, ii] = self.asleg(ii, theta, phi_0)
                        
            # Allocate derivative of associated Legendre polynomials
            dPlk[:, ii] = self.deriv_asleg(ii, theta, phi_0, Plk[:, ii])
        
        self.A = A
        self.normA = normA
        self.dPlk = dPlk
        self.Plk = Plk
        
        if self.case == 'antenna':    
            # Generate Spherical Harmonics Matrix for antenna measurements
            # It should be noted that in antenna measurement, DC component will not be
            # considered
            
            self.A = self.A[:, 1::]
            self.normA = self.normA[:, 1::]
            self.dPlk = self.dPlk[:, 1::]
            self.Plk = self.Plk[:, 1::]
            self.N = self.N - 1
            self.deg_order = self.deg_order[1::, :]
        
      
class MatrixWigner(Matrix):
    
    """ 
    Class to construct matrix from Wigner D-functions
   
    Attributes
    -------
    A : ndarray
        Matrix from Wigner D-functions
    normA : ndarray
        Normalized matrix from Wigner D-functions
        w.r.t l2-norm
     
    Wignerd : ndarray
        Matrix from Wigner (small) d-functions, note it is different with Wigner D-functions
    
    dWignerd : ndarray
        Matrix derivative of Wigner (small) d-functions
    
    m : int
        row dimension of the matrix (size samples)
    N : int
        col dimension of the matrix (combination of degree and orders)
    deg_order : ndarray
        combination of degre and order
    """        
    
    def degree_orders(self):
        
        '''
        Generating combination of degree and orders for Wigner D-function
        '''
        
        self.N = self.B*(2*self.B-1)*(2*self.B+1)//3
        lkn = np.zeros((self.N, 3), dtype=self.float_dtype)
        idx_beg = 0
        for l_deg in range(self.B):
            n = range(-l_deg, l_deg + 1)
            k = range(-l_deg, l_deg + 1)
            mesh_k_n = np.array(np.meshgrid(k, n))
            k_n = mesh_k_n.T.reshape(-1, 2)
            idx = len(n)**2
            idx_end = idx_beg + idx-1
            lkn[idx_beg:idx_end + 1, 0] = np.full((1, idx), l_deg)
            lkn[idx_beg:idx_end + 1, 1:] = k_n
            idx_beg = idx_beg + idx
        self.deg_order = lkn           
    
    def param(self, ii):
        
        """
        Method to generate parameters for matrix Wigner D-functions
        Since the degree and orders for Jacobi polynomials is different
        see literature to construct Wigner D-functions
        """
        
        # Set initial parameters
      
        if self.deg_order[ii, 2] >= self.deg_order[ii, 1]:
            eta = 1
        else:
            eta = (-1)**(self.deg_order[ii, 2] - self.deg_order[ii, 1])

        # Set Normalization (SO(3) normalization)
        normalization = np.sqrt((2.0*self.deg_order[ii, 0]+1)/(8.0*np.pi**2))
        
        # Parameters degree and orders for Jacoby polynomials
        self.mu_plus = np.abs(self.deg_order[ii, 1] - self.deg_order[ii, 2])
        self.vu_plus = np.abs(self.deg_order[ii, 1] + self.deg_order[ii, 2])
        self.s_plus = self.deg_order[ii, 0] - (self.mu_plus + self.vu_plus)/2.0
        
        norm_gamma = np.sqrt((math.factorial(self.s_plus)*math.factorial(self.s_plus +
                                                                         self.mu_plus +
                                                                         self.vu_plus)) /
                             (math.factorial(self.s_plus + self.mu_plus) *
                             (math.factorial(self.s_plus + self.vu_plus))))
        
        self.weight = eta*normalization*norm_gamma
    
    def wigner_d(self, theta):
        
        """
        A method to calculate Wigner d-functions
        """
        
        # Generate Wigner d-functions

        return (self.weight*(np.sin(theta/2.0)**self.mu_plus)*(np.cos(theta/2.0)**self.vu_plus) *
                (Plkn(self.s_plus, self.mu_plus, self.vu_plus, np.cos(theta))))
    
    def deriv_wigner(self, theta, wignerd, normalize):
        """
        A method to calculate derivative of Wigner d-functions
        """
        # Jacobi polynomials
        
        Jacobi_last = (Plkn(self.s_plus - 1, self.mu_plus + 1, self.vu_plus + 1, np.cos(theta)) /
                       normalize)
        
        # Derivative
        
        wignerd_deriv = (((self.mu_plus*np.sin(theta))/(2.0*(1 - np.cos(theta))) -
                          (self.vu_plus*np.sin(theta))/(2.0*(1 + np.cos(theta))))*wignerd -
                         (np.sin(theta)*self.weight*(self.mu_plus + self.vu_plus + self.s_plus + 1) 
                         * 0.5 * (np.sin(theta/2.0)**self.mu_plus)*(np.cos(theta/2.0)**self.vu_plus)
                         * (Jacobi_last)))   
        
        return wignerd_deriv
    
    def gen_matrix(self):
        
        """
        Method to generate Wigner D-functions matrix and derivative of Wigner d-functions
        """
    
        # Generate degree and order
        self.degree_orders()
    
        # Generate matrix
        A = np.zeros((self.m, self.N), dtype=self.complex_dtype)
        normA = np.zeros_like(A)
        dwignerd = np.zeros((self.m, self.N), dtype=self.float_dtype)
        wignerd = np.zeros_like(dwignerd)
    
        theta = self.angles['theta']
        phi = self.angles['phi']
        chi = self.angles['chi']
        
        # Asign phi, chi = 0 to get Wigner (small) d functions
        # phi_0 = np.zeros(self.m, dtype=self.float_dtype)
        # chi_0 = np.copy(phi_0)
        
        for ii in range(self.N):
             
            # Parameters
            self.param(ii=ii)
        
            # Generate Wigner D-functions sensing matrix and their (column) normalization
       
            A[:, ii] = (np.exp(-1j*self.deg_order[ii, 1]*phi) *
                        self.wigner_d(theta) *
                        np.exp(-1j*self.deg_order[ii, 2]*chi))
            
            # Normalize w.r.t l2 norm
            normA[:, ii] = A[:, ii]/LA.norm(A[:, ii])
            
            wignerd[:, ii] = self.wigner_d(theta)/LA.norm(A[:, ii])
            
            # Calculate derivative of Wigner d-functions
                
            dwignerd[:, ii] = self.deriv_wigner(theta=theta, wignerd=wignerd[:, ii],
                                                normalize=LA.norm(A[:, ii]))
 
        self.A = A
        self.normA = normA
        self.wignerd = wignerd
        self.dwignerd = dwignerd
        
        if self.case == 'antenna':

            # Generate Spherical Harmonics Matrix for antenna measurements
            # It should be noted that in antenna measurement, DC component will not be
            # considered
        
            self.A = self.A[:, 1::]
            self.normA = self.normA[:, 1::]
            self.wignerd = self.wignerd[:, 1::]
            self.dwignerd = self.dwignerd[:, 1::]
            self.N = self.N - 1
            self.deg_order = self.deg_order[1::, :]
  
   
class MatrixSNF(Matrix):    
     
    """ 
    Class to construct matrix from Wigner D-functions
    for spherical near-field (SNF)
      
    
    
    Attributes
    -------
    A : ndarray
        Matrix from Wigner D-functions
        
    normA : ndarray
        Normalized matrix from Wigner D-functions
        w.r.t l2-norm
     
    dmm_plus1, dmm_plus2
    dmm_min1, dmm_min2 : ndarray
        Matrix combination of Wigner (small) d-functions, specifically only for spherical near-field measurements
        (see. paper in [])
    
    d_dmm_plus1, d_dmm_plus2
    d_dmm_min1, d_dmm_min2 : ndarray
        Derivative of matrix combination defined above 
        
    m : int
        row dimension of the matrix (size samples)
    N : int
        col dimension of the matrix (combination of degree and orders)
    deg_order : ndarray
        combination of degre and order
    
    """
    
    def degree_orders_snf(self):
        
        '''
        Generating combination of degree and orders for Wigner D-function in
        spherical near-field measurements (SNF)
        '''
        self.N = 2*self.B**2 + 4*self.B
        lk = np.zeros((self.N//2,2), dtype=self.float_dtype)
        idx_beg = 0
        for l in np.arange(1,self.B + 1):
            k = range(-l,l+1)
            idx = len(k)
            idx_end = idx_beg + idx-1
            lk[idx_beg:idx_end+1,1] = k
            lk[idx_beg:idx_end+1,0] = np.full((1,idx),l)
            idx_beg = idx_beg + idx
        self.deg_order = lk    
       
    
    def param(self, mu, ii):
        
        """
        Method to generate parameters for matrix Wigner D-functions
        Since the degree and orders for Jacobi polynomials is different
        see literature to construct Wigner D-functions
        """
 
        ## Parameters
        normalization = math.sqrt((2.0*self.deg_order[ii,0]+1)/(8.0*np.pi**2))
            
            
        if mu >= self.deg_order[ii,1]:
            eta = 1
        else:
            eta = (-1)**(mu - self.deg_order[ii,1])
        
        ## Set Wigner d for positive
        self.mu_sign = np.abs(self.deg_order[ii,1] - mu)
        self.vu_sign = np.abs(self.deg_order[ii,1] + mu)
        self.s_sign = self.deg_order[ii,0] - (self.mu_sign + self.vu_sign)/2.0
        
        norm_gamma = np.sqrt((math.factorial(self.s_sign)*math.factorial(self.s_sign + self.mu_sign + self.vu_sign))/
                             (math.factorial(self.s_sign + self.mu_sign)*(math.factorial(self.s_sign + self.vu_sign))))
            
        self.weight_sign = normalization*norm_gamma*eta    
    
     
    
    def wigner_d(self, theta, mu, ii):
        
        """
        Method to generate Wigner d-functions
        """
        self.param(mu,ii)
        
        wigner_d = (self.weight_sign*(np.sin(theta/2)**self.mu_sign)*(np.cos(theta/2)**self.vu_sign)*
                             Plkn(self.s_sign,self.mu_sign,self.vu_sign,np.cos(theta)))
        return wigner_d
    
    def deriv_wigner(self, theta, dmm, mu, ii):
        
        """
        Method to generate derivative Wigner d-functions
        """
        self.param(mu,ii)
        
        ## Jacobi polynomials
        Jacobi_last_sign = (Plkn(self.s_sign - 1, self.mu_sign + 1, self.vu_sign + 1, np.cos(theta)))

        
        ## derivative
        wignerd_deriv_sign = (((self.mu_sign*np.sin(theta))/(2.0*(1 - np.cos(theta))) -
                               (self.vu_sign*np.sin(theta))/(2.0*(1 + np.cos(theta))))*dmm -
                              (np.sin(theta)*self.weight_sign*(self.mu_sign + self.vu_sign + self.s_sign + 1)*0.5*
                               (np.sin(theta/2.0)**self.mu_sign)*(np.cos(theta/2.0)**self.vu_sign)*
                               (Jacobi_last_sign)))
        
        return wignerd_deriv_sign
    
          

    
    
    def gen_matrix(self):
        
        """
        Method to generate Wigner D-functions matrix for SNF and its derivative on theta
        
        """
        
        #### Generate degree and order
        self.degree_orders_snf()
                

        ### Alocate the matrix
        Basis_1 = np.zeros((self.m, self.N//2), dtype = self.complex_dtype)
        Basis_2 = np.copy(Basis_1)
        
        norm_Basis_1 = np.copy(Basis_1)
        norm_Basis_2 = np.copy(Basis_2)
        
        dmm_plus = np.zeros((self.m, self.N//2), dtype = self.float_dtype)
        dmm_min = np.zeros((self.m, self.N//2), dtype = self.float_dtype)

        dmm_plus1 = np.zeros((self.m, self.N//2), dtype = self.float_dtype)
        dmm_plus2 = np.zeros_like(dmm_plus1)
        
        dmm_min1  = np.zeros((self.m, self.N//2), dtype = self.float_dtype)
        dmm_min2  = np.zeros_like(dmm_min1)
        
        
        ### allocate derivative
        d_dmm_plus = np.zeros((self.m, self.N//2), dtype = self.float_dtype)
        d_dmm_min = np.zeros_like(d_dmm_plus)
        
        
        d_dmm_plus1 = np.zeros((self.m, self.N//2), dtype = self.float_dtype)
        d_dmm_plus2 = np.zeros_like(d_dmm_plus1)
        
        d_dmm_min1 = np.zeros((self.m, self.N//2), dtype = self.float_dtype)
        d_dmm_min2 = np.zeros_like(d_dmm_min1)
        
        ## Angles
        theta = self.angles['theta']
        phi = self.angles['phi']
        chi = self.angles['chi']

        #####
     

        for ii in range(self.N//2):
            
            
            ## Wigner d plus
            dmm_plus[:,ii] = self.wigner_d(theta = theta, mu = 1, ii = ii)

            
            ## Derivative Wigner d plus w.r.t theta
            d_dmm_plus[:,ii] = self.deriv_wigner(theta = theta, dmm = dmm_plus[:,ii], mu = 1, ii = ii) 
 
            ## Wigner d min
            dmm_min[:,ii] = self.wigner_d(theta = theta, mu = -1, ii = ii)
            
            ## Derivative Wigner d negative w.r.t theta
            d_dmm_min[:,ii] = self.deriv_wigner(theta = theta, dmm = dmm_min[:,ii], mu = -1, ii = ii) 
            
            
            ## Generate Wigner D for s = 1 (TE)
            Basis_1[:,ii] = (np.exp(1j*chi)*dmm_plus[:,ii] + np.exp(-1j*chi)*
                             dmm_min[:,ii])*np.exp(1j*self.deg_order[ii,1]*phi)
            
            ## Normalizing w.r.t l2 norm
            norm_Basis_1[:,ii] = Basis_1[:,ii]/LA.norm(Basis_1[:,ii])
            
            
            ## Generate Wigner D for s = 2 (TM)
            Basis_2[:,ii] = (np.exp(1j*chi)*dmm_plus[:,ii] - np.exp(-1j*chi)*
                             dmm_min[:,ii])*np.exp(1j*self.deg_order[ii,1]*phi)
            
            ## Normalizing w.r.t l2 norm
            norm_Basis_2[:,ii] = Basis_2[:,ii]/LA.norm(Basis_2[:,ii])

            ### Store normalization
            dmm_plus1[:,ii]    =   dmm_plus[:,ii]/LA.norm(Basis_1[:,ii])
            d_dmm_plus1[:,ii]  = d_dmm_plus[:,ii]/LA.norm(Basis_1[:,ii])
            dmm_min1[:,ii]     =   dmm_min[:,ii]/LA.norm(Basis_1[:,ii])
            d_dmm_min1[:,ii]   = d_dmm_min[:,ii]/LA.norm(Basis_1[:,ii])
                                                       
                                                       
            dmm_plus2[:,ii]    =   dmm_plus[:,ii]/LA.norm(Basis_2[:,ii])
            d_dmm_plus2[:,ii]  = d_dmm_plus[:,ii]/LA.norm(Basis_2[:,ii])
            dmm_min2[:,ii]     =   dmm_min[:,ii]/LA.norm(Basis_2[:,ii])
            d_dmm_min2[:,ii]   = d_dmm_min[:,ii]/LA.norm(Basis_2[:,ii])                                           
            
            
           

        ### Concatenate to get matrix for SNF
        A =  np.concatenate((Basis_1,Basis_2),axis = 1)
        normA = np.concatenate((norm_Basis_1, norm_Basis_2), axis = 1)
 
        
        self.A = A
        self.normA = normA
        self.d_dmm_plus1 = d_dmm_plus1
        self.d_dmm_plus2 = d_dmm_plus2
                                                       
        self.d_dmm_min1 = d_dmm_min1  
        self.d_dmm_min2 = d_dmm_min2
                              
        self.dmm_plus1 = dmm_plus1
        self.dmm_plus2 = dmm_plus2
                                                       
        self.dmm_min1 = dmm_min1
        self.dmm_min2 = dmm_min2
        
       