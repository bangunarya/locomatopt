import numpy as np 

def vector_coherence_sh(params_mat, mat):
    '''
    Function to vectorizing the coherence for spherical harmonics sensing matrix
    
    Parameters
    ----------
    
    params_mat : dict
        Parameters to construct specific matrix with parameters bandwidth (B)
        column dimension (N), type of matrix (types), combination degree and orders (col_comb),
        spherical near-field cas or general expansion (case)
            
    mat : dict
        Construction of spherical harmonics matrix
        
        
    Returns
    -------
    vect_comb :array
        vectorizing coherence of spherical harmonics sensing matrix
    '''
    
    col_comb = params_mat['col_comb']
        
    ## Combination of degree and orders
    comb_lk = [mat.deg_order[col_comb[:,0],:],
               mat.deg_order[col_comb[:,1],:]]
        
    ## Product of combination of degree and order associated Legendre 
    ProductasLeg = (mat.Plk[:,col_comb[:,0]]*
                    mat.Plk[:,col_comb[:,1]])
        
    ## Differences order
    k = comb_lk[0][:,1] - comb_lk[1][:,1]
   

    ## Allocation
    phi = mat.angles['phi']
        
               
    mat_cos = np.cos(np.outer(phi,k))
    mat_sin = np.sin(np.outer(phi,k))
        
    ## Product of function for all combination degree and order 
    vect_comb = np.sqrt(np.abs(np.sum(ProductasLeg*mat_cos,0))**2 +
                        np.abs(np.sum(ProductasLeg*mat_sin,0))**2)
    return vect_comb

def vector_coherence_wigner(params_mat, mat):
    '''
    Function to vectorizing the coherence for Wigner D-functions sensing matrix
    
    Parameters
    ----------
    
    params_mat : dict
        Parameters to construct specific matrix with parameters bandwidth (B)
        column dimension (N), type of matrix (types), combination degree and orders (col_comb),
        spherical near-field cas or general expansion (case)
            
    mat : dict
        Construction of Wigner D-functions sensing matrix
        
        
    Returns
    -------
    vect_comb :array
        vectorizing coherence of  Wigner D-functions sensing matrix
    '''
    
    col_comb = params_mat['col_comb']
         
    ### Combination for coherence
    comb_lkn = [mat.deg_order[col_comb[:,0],:], mat.deg_order[col_comb[:,1],:]]
        
    k = comb_lkn[0][:,1] - comb_lkn[1][:,1]
    n = comb_lkn[0][:,2] - comb_lkn[1][:,2]

    ## Product of combination degree and orders Wigner d-functions
    ProductWignerd = (mat.wignerd[:,col_comb[:,0]]*
                      mat.wignerd[:,col_comb[:,1]])

    ## Allocation
     
    phi = mat.angles['phi']
    chi = mat.angles['chi']
         

    mat_cos = np.cos(np.outer(phi,k) + np.outer(chi,n))
    mat_sin = np.sin(np.outer(phi,k) + np.outer(chi,n))
        
    vector_comb = np.sqrt(np.abs(np.sum(ProductWignerd*mat_cos,0))**2 +
                          np.abs(np.sum(ProductWignerd*mat_sin,0))**2) 
    return vector_comb
    

def vector_coherence_snf(params_mat, mat):
    
    '''
    Function to vectorizing the coherence for Wigner D-functions sensing matrix
    for spherical near-field measurements (SNF)
    
    Parameters
    ----------
    
    params_mat : dict
        Parameters to construct specific matrix with parameters bandwidth (B)
        column dimension (N), type of matrix (types), combination degree and orders (col_comb),
        spherical near-field cas or general expansion (case)
            
    mat : dict
        Construction of Wigner D-functions sensing matrix
        for spherical near-field measurements (SNF)
        
        
    Returns
    -------
    vect_comb :array
        vectorizing coherence of  Wigner D-functions sensing matrix
        for spherical near-field measurements (SNF)
    '''
    col_comb = params_mat['col_comb']
        
    N = mat.N//2
      
    norm_A1 = mat.normA[:,0:N]
    norm_A2 = mat.normA[:,N::]
        
    #### Combination    
    idx_12 = np.nonzero(col_comb[:,1] > col_comb[:,0])[0]

        
    ## ProductCoh3
    ProductCoh3 = norm_A1[:,col_comb[:,0]]*np.conj(norm_A2[:,col_comb[:,1]])
    ProductCoh1 = norm_A1[:,col_comb[idx_12,0]]*np.conj(norm_A1[:,col_comb[idx_12,1]])
    ProductCoh2 = norm_A2[:,col_comb[idx_12,0]]*np.conj(norm_A2[:,col_comb[idx_12,1]])
       
       
       
    ##
    ProductCohTot = np.concatenate((np.sum(ProductCoh1,0), 
                                    np.sum(ProductCoh3,0), 
                                    np.sum(ProductCoh2,0)), axis = 0)


    vector_comb = np.abs(ProductCohTot) # np.abs(np.sum(ProductCoh,0))
      
    return vector_comb