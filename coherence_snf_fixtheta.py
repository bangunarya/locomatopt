import matplotlib.pyplot as plt
import numpy as np
import os
import json
from new_code.sampling import SpiralSampling, UniformRandom, PoleRandom
from new_code.metric import params_matrix
from new_code.gradientdescent import GradDescent
from new_code.adam import Adam
from new_code.adagrad import AdaGrad
from new_code.adadelta import AdaDelta
from new_code.auglmethod import ALM
from new_code.matrix import MatrixSH, MatrixWigner, MatrixSNF
from new_code.metric import coherence
from numpy import linalg as LA
import copy



############################ Parameters ################################
## Bandwidth
B = 16

## Antenna or normal expansion
case = 'antenna'

## Sphere or Rotation Group
types = 'snf'
## Generate dimension
N, col_comb = params_matrix(types = types, B = B, case = case)
## Generate sampling points
m = np.arange(17, N - 45, 45).astype(np.int64)  # Samples
 
## Matrix Parameters
params_mat = {'B':B, 'N':N, 'types':types, 
              'col_comb':col_comb, 'case':case}
p = 9
## Gradient Parameters
params_grad = {'max_iter':  500 , 'eps': 1e-6, 
               'update': 'fix_theta', 'p_norm': p}

print(N)


## Path
folder = 'result'
path = os.path.join(os.getcwd(),folder)
filename = types + '_' + case + '_' + params_grad['update'] + '_alm'

############################# Main Loop ################################
MC = 1 ## Monte carlo

## Allocation angles for samples
alm_ang_sample = []
gd_ang_sample = [] 

## Allocation coherence for samples
alm_coh_sample = []
gd_coh_sample = [] 

## Loop for samples
for idx in range(len(m)):
    
    ### Generate samples
    
    unif_rand = UniformRandom(m = m[idx], polar = 'snf') # Wigner polar should random
        
        
    ########################### Gradient ########################
    gradesc = GradDescent(params_mat = params_mat, params_grad = params_grad)
    res_gradesc = gradesc.run_algo(angles = unif_rand.angles)
    
    select_mat = {'sh'    : MatrixSH,
                  'wigner': MatrixWigner,
                  'snf'   : MatrixSNF}
    
    mat_gd = select_mat[params_mat['types']](B = params_mat['B'], angles =  res_gradesc['angle'],
                                             case = params_mat['case'])
        
        
    print('Sample (m): ',m[idx],' GD ', res_gradesc['coherence'], coherence(mat_gd.normA))
        
    gd_ang_sample.append(res_gradesc['angle'])
    gd_coh_sample.append(res_gradesc['coherence'])

    ########################### ALM ##############################
    alm = ALM(params_mat = params_mat, params_grad = params_grad)
    res_alm = alm.run_algo(angles = unif_rand.angles)
    
    select_mat = {'sh'    : MatrixSH,
                  'wigner': MatrixWigner,
                  'snf'   : MatrixSNF}
        
    mat_alm = select_mat[params_mat['types']](B = params_mat['B'], angles =  res_alm['angle'],
                                                  case = params_mat['case'])
       
        
    print('Sample (m): ',m[idx],' ALM;', res_alm['coherence'], coherence(mat_alm.normA))
        
    alm_ang_sample.append(res_alm['angle'] )
    alm_coh_sample.append(res_alm['coherence'] )

    

total_result = {'m':m, 'B':B, 'N':N,
                'alm_ang' : alm_ang_sample, 'alm_coh' : alm_coh_sample,
                'gd_ang' : gd_ang_sample, 'gd_coh' : gd_coh_sample}


## Store Files
os.makedirs(path,exist_ok = True)
path_file = os.path.join(path,filename + '.npy')
np.save(path_file,total_result)
