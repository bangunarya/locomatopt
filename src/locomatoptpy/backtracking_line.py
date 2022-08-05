

def backtrack_line_search(first_condition, second_condition):
    """
    Using Armijo condition
    f( x + alpha * (-f_grad(x)) ) < f(x) + c * alpha * f_grad(x) * (-f_grad(x)) 
    In this case cond1 < cond2. Initial data 0 < c < 0.5 (typical:10^-4 0) < rho <= 1
    
    """
    alpha = 1
    rho = 0.8
    c = 1e-4
    
    while (first_condition(alpha) - second_condition(alpha*c)) >= 1e-4:
        
        alpha *= rho
    
    return alpha
