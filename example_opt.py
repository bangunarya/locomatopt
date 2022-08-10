import numpy as np
import os
from locomatopt.coherence_optimization import coherence_optimization_grad_descent 

if __name__ == '__main__':
    
    mode = 'normal'
    basis = 'sh'
    max_iter = 400
    bandwidth = 10
    eps = 1e-6
    update = 'fix_theta'
    p_norm = 9
    samples = np.arange(21, 100, 8)

    # Run algorithms
    total_result_gd = coherence_optimization_grad_descent(bandwidth=bandwidth, mode=mode, 
                                                          basis=basis,
                                                          max_iter=max_iter, eps=eps, 
                                                          update=update,
                                                          p_norm=p_norm, samples=samples)

    # Path
    folder = os.path.join('results/', mode, basis)
    path = os.path.join(os.getcwd(), folder)
    filename = basis + '' + mode + '' + update 

    # Store Files
    os.makedirs(path, exist_ok=True)
    path_file = os.path.join(path, filename + '.npy')
    np.save(path_file, total_result_gd)
