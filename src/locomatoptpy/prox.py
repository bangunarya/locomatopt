import numpy as np

def project_l1_ball(vector, radius , stop_thr):
    
    """ 
    This code reimplemented from the project_l1_ball.m matlab file from  stackexchange,
    see the references
    
    A method to solve the Orthoginal Porjection Problem of the input vector onto the
    L1 Ball using Dual Function and Newton Iteration.
    
        Proximal Operator of Infinity Norm:

        Proxtkλg(⋅)(v)=Proxλ∥⋅∥∞(v)=v−λProj{∥⋅∥1≤1}(vλ)
    
        Since dual norm of infinity norm is l1-norm, then 
        we project to the l1 norm ball.
        See [] for more explanation
    
        This is pure reimplementation matlab file from []
    
        Input : 
        vector    -   Input Vector.
                
        radius    -   Ball Radius.
                      Sets the Radius of the L1 Ball. For Unit L1 Ball
                      set to 1.
        stop_thr  -   Stopping Threshold.
                      Sets the trheold of the Newton Iteration. The
                      absolute value of the Objective Function will be
                      below the threshold.
        Output:
        vect_out  -   Output Vector.
                      The projection of the Input Vector onto the L1
                      Ball.
     
        References
        https://math.stackexchange.com/questions/2327504.

        """
     
    if (np.sum(np.abs(vector)) <= radius):
        return vector
        
    else:
        paramLambda = 0
            
            # The objective functions which its root (The 'paramLambda' which makes it
            # vanish) is the solution
        
            # Set less than zero to zero
        subs = (np.abs(vector) - paramLambda) 
        subs[subs < 0] = 0        
        
        
        objVal = np.sum(subs) - radius

        while(np.abs(objVal) > stop_thr):
                
                # Set less than zero to zero
        
            subs = (np.abs(vector) - paramLambda)
            subs[subs < 0] = 0
            
            
            objVal  = np.sum(subs) - radius
                
                # Derivative of 'objVal' with respect to Lambda
            df              = np.sum(-1*((np.abs(vector) - paramLambda) > 0))
                
               # Newton Iteration
            paramLambda     = paramLambda - (objVal / df)

           
            # Enforcing paramLambda >= 0. Otherwise it suggests || vY ||_1 <= radius.
            # Hence the Optimal vX is given by vX = vY.
           
        paramLambda = max(paramLambda,0)
        
        
        ## Return value
        temp_sub = np.abs(vector) - paramLambda
        temp_sub[temp_sub < 0] = 0
            
        return (vector/np.abs(vector))*temp_sub   