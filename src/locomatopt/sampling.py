import numpy as np


class SamplingPoints:
 
    """ 
    Base class of sampling points on the sphere and rotation group.
    In this case we have:
    - Equiangular sampling
    - Spiral sampling [1,6]
    - Hammersley sampling [2,6]
    - Random sampling w.r.t specific measure in [3,4]
    - Uniform random sampling [5]
    
      
    Parameters
    ----------
    m : int
        number of samples
    space: str
        choose to generate sampling points on the sphere or rotation group.
        the differences we have another variable polarization (chi)
    
    sampling: str
        choose specific sampling points
        
    Attributes
    ---------
    theta : ndarray
            sampling points on the elevation [0,pi]
    phi   ; ndarray
            sampling points on the azimuth [0,2.pi)
    chi   : ndarray
            sampling points on the polarization[0,2.pi)
    

    References:
    ----------
    [1] Saff, Edward B., and Amo BJ Kuijlaars. "Distributing many points on a sphere." 
        The mathematical intelligencer 19.1 (1997): 5-11.
    [2] Equidistribution on the sphere, J Cui, W Freeden - SIAM Journal on Scientific Computing,
    [3] Sparse recovery for spherical harmonic expansions, H Rauhut, R Ward 
    [4] Burq, Nicolas, et al. "Weighted eigenfunction estimates with applications to 
        compressed sensing" SIAM Journal on Mathematical Analysis 44.5 (2012): 3481-3501.
    [5] https://mathworld.wolfram.com/SpherePointPicking.html
    [6] A Comparison of Popular Point Configurations on S2
        D.P. Hardin, T. J. Michaels, E.B. Saff

    """
    
    def __init__(self, m, basis):
       
        self.m = m
        self.basis = basis
         
    def generate_angles(self):
        """
        Method to generate total sampling points
        """
          
        angles = {'theta': self.theta(),
                  'phi': self.phi()}

        if self.basis == 'wigner' or self.basis == 'snf':
            angles['chi'] = self.chi()

        return angles

    def theta(self):
        """
        Method to generate samples on theta
        """
        pass
    
    def phi(self):
        
        """
        Method to generate samples on phi
        """
        pass
        
    def chi(self):
        
        """
        A method to generate polarization
        """
        pass

    
class EquiangularSampling(SamplingPoints):
    
    """ 
    Class to generate equiangular sampling points
    """  
    
    def theta(self): 
        """
        Method to generate theta        
        """
 
        return np.linspace(0, np.pi, self.m)
        
    def phi(self):
        
        """
        Method to generate phi 
        
        """
        
        return np.linspace(0, 2.0*np.pi, self.m)
    
    def chi(self):
        """
        Method to generate chi deterministic
        """

        return np.linspace(0, 2*np.pi, self.m)


class SpiralSampling(SamplingPoints):
    
    """
    Class to generate spiral sampling points
    """   

    def theta(self):
        """
        Method to generate theta
        """
         
        l_range = np.arange(self.m)
        theta = np.arccos(-1 + (2.0*l_range)/(self.m-1))
                
        return theta 
    
    def phi(self):
        """
        Method to generate phi
        """
        
        C = 3.6
        phi = np.zeros(self.m)
        for l_idx in range(1, self.m-1):
            h = -1 + (2.0*l_idx)/(self.m-1)
            phi[l_idx] = (phi[l_idx-1] + (C/np.sqrt(self.m))*(1/np.sqrt(1-h**2.0))) % (2.0*np.pi)
        
        return phi

    def chi(self):
        """
        Method to generate chi deterministic
        """

        return np.linspace(0, 2*np.pi, self.m)


class FibonacciSampling(SamplingPoints):
    
    """
    Class to generate Fibonacci sampling points
    """
    
    def theta(self):
        """
        Method to generate theta
        """
        
        N = int(np.ceil((self.m-1)/2.0))  # Has to be an odd number of points.
        theta = np.zeros(2*N + 1)

        k = 0
        for ii in range(-N, N + 1):
            lat = np.arcsin(2.0*ii/(2.0*N + 1))
            theta[k] = np.pi/2 - lat
            k += 1
        
        return theta
    
    def phi(self):
        """
        Method to generate phi
        """
        
        N = int(np.ceil((self.m-1)/2.0))  # Has to be an odd number of points.
        
        phi = np.zeros(2*N + 1)
        gr = (1 + np.sqrt(5))/(2.0)
        k = 0
        for ii in range(-N, N + 1):
            lat = np.arcsin(2.0*ii/(2.0*N + 1))
            lon = 2.0*np.pi*ii/gr            
            phi[k] = np.arctan2(np.cos(lat)*np.sin(lon), np.cos(lat)*np.cos(lon))
            if phi[k] < 0:
                phi[k] = phi[k] + 2.0*np.pi

            k += 1
        return phi
    
    def chi(self):
        """
        Method to generate chi
        """
        N = int(np.ceil((self.m-1)/2.0))  # Has to be an odd number of points.
         
        return np.linspace(0, 2*np.pi, 2*N+1)


class HammersleySampling(SamplingPoints):
    """
    Class to generate Hammersley sampling points
    """
    
    def basexpflip(self, k, b): 
        """
        reversed base-b expansion of positive integer k
        """
        j = int(np.fix(np.log(k)/np.log(b))) + 1
        a = np.zeros(j)
        q = b**(j-1)
        for ii in range(j):
            a[ii] = int(np.floor(k/q))
            k = k - q*a[ii]
            q = q/b
        a = a[::-1]
        return a
    
    def vdcorput(self, k, b):
        
        """
        Method to generate van der corput sequence
        
        """
        s = np.zeros(k)
        for i in np.arange(k)+1:
            a = self.basexpflip(i, b)
            g = np.power(b, np.arange(len(a))+1)
            s[i-1] = np.sum(np.divide(a, g))
        return s
     
    def theta(self):
        """
        Method to generate theta
        """
        
        # Generate Hammersley points
        t = self.vdcorput(self.m, 2)
        t = 2*t - 1
        
        return np.arccos(t)
    
    def phi(self):
        
        """
        Method to generate phi
        """
        
        phi = 2*np.pi*((2*(np.arange(self.m)+1)-1)/2.0/self.m)
        phi[phi < 0] = phi[phi < 0] + 2.0*np.pi
        
        return phi

    def chi(self):
        """
        Method to generate chi deterministic
        """

        return np.linspace(0, 2*np.pi, self.m)


class PoleRandom(SamplingPoints):
    """
    Class to generate random sampling points
    according to [3]
    """
    
    def theta(self):
        
        """
        Method to generate theta
        """    
        return np.random.rand(self.m)*np.pi
        
    def phi(self):
        """
        Method to generate phi
        """
        
        return np.random.rand(self.m)*2.0*np.pi
    
    def chi(self):
        """
        Method to generate chi randomly or w.r.t snf
        """
       
        if self.basis == 'snf':
            chi = (np.arange(self.m) % 2)*(np.pi/2)
        if self.basis == 'wigner':
            chi = np.random.rand(self.m)*2.0*np.pi
                   
        return chi


class UniformRandom(SamplingPoints):       
    """
    Class to generate random sampling points
    according to [5]
    """
        
    def theta(self):
        """
        Method to generate theta
        """
        
        return np.arccos(2.0*np.random.rand(self.m) - 1)
        
    def phi(self):
        """
        Method to generate phi
        """
        return np.random.rand(self.m)*2.0*np.pi
      
    def chi(self):
        """
        Method to generate chi randomly or w.r.t snf
        """
       
        if self.basis == 'snf':
            chi = (np.arange(self.m) % 2)*(np.pi/2)
        if self.basis == 'wigner':
            chi = np.random.rand(self.m)*2.0*np.pi
                   
        return chi