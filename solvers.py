import numpy as np
import time

class IterativeSolvers:
    """
    Collection of iterative methods to solve the Linear System Ax = b
    derived from the discretized Poisson Equation on a 2D grid.
    
    The linear algebra relation is:
    A * u = f
    
    Where 'A' is the discrete Laplacian operator.
    """

    @staticmethod
    def jacobi(u_initial, f_source, tol=1e-4, max_iter=5000):
        """
        Solves using Jacobi Method.
        Algebraic splitting: A = D - (L + U)
        Update formula: D * x^(k+1) = (L + U) * x^(k) + b
        
        In grid terms: The new value is the average of the old neighbors.
        """
        print(f"--- Starting Jacobi Method ---")
        start_time = time.time()
        
        u = u_initial.copy()
        u_new = u.copy()
        residuals = []
        
        # Matrix dimensions (Grid N x N)
        N = u.shape[0]

        for k in range(max_iter):
            # Vectorized update (Linear Algebra: Simultaneous updates)
            # u[i,j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - h^2*f[i,j])
            u_new[1:-1, 1:-1] = 0.25 * (u[0:-2, 1:-1] + 
                                        u[2:, 1:-1] + 
                                        u[1:-1, 0:-2] + 
                                        u[1:-1, 2:] - 
                                        f_source[1:-1, 1:-1])
            
            # Calculate Residual (Norm of the difference vector)
            # || r || = || u_new - u_old ||
            diff = np.linalg.norm(u_new - u)
            residuals.append(diff)
            
            # Prepare for next iteration
            u = u_new.copy()
            
            if diff < tol:
                print(f"Jacobi converged at iteration {k}. Time: {time.time()-start_time:.4f}s")
                return u, residuals
        
        print("Jacobi did not converge.")
        return u, residuals

    @staticmethod
    def gauss_seidel(u_initial, f_source, tol=1e-4, max_iter=5000):
        """
        Solves using Gauss-Seidel Method.
        Algebraic splitting: A = (D - L) - U
        Update formula: (D - L) * x^(k+1) = U * x^(k) + b
        
        In grid terms: We use the NEWEST available neighbor values immediately.
        """
        print(f"--- Starting Gauss-Seidel Method ---")
        start_time = time.time()
        
        u = u_initial.copy()
        residuals = []
        N = u.shape[0]

        for k in range(max_iter):
            u_old_norm = np.linalg.norm(u) # Store norm to compute diff later
            
            # Iterating explicitly to enforce using the NEW values (Sequential dependency)
            # Note: This is slower in Python than C++, but necessary to show the logic.
            for i in range(1, N-1):
                for j in range(1, N-1):
                    u[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + 
                                      u[i, j-1] + u[i, j+1] - 
                                      f_source[i, j])
            
            # Calculate error norm
            # Note: To save time in python loops, we approximate error check
            # or we can do a full norm check.
            u_current_norm = np.linalg.norm(u)
            diff = abs(u_current_norm - u_old_norm)
            residuals.append(diff)
            
            if diff < tol:
                print(f"Gauss-Seidel converged at iteration {k}. Time: {time.time()-start_time:.4f}s")
                return u, residuals
                
        return u, residuals

    @staticmethod
    def sor(u_initial, f_source, omega, tol=1e-4, max_iter=5000):
        """
        Solves using Successive Over-Relaxation (SOR).
        Algebra concept: Introduce a relaxation parameter 'omega' to extrapolate
        the Gauss-Seidel direction.
        
        x^(k+1) = (1 - omega) * x^(k) + omega * x_GaussSeidel
        """
        print(f"--- Starting SOR Method (omega={omega}) ---")
        start_time = time.time()
        
        u = u_initial.copy()
        residuals = []
        N = u.shape[0]

        for k in range(max_iter):
            u_old_norm = np.linalg.norm(u)
            
            for i in range(1, N-1):
                for j in range(1, N-1):
                    # 1. Compute Gauss-Seidel prediction
                    gs_prediction = 0.25 * (u[i-1, j] + u[i+1, j] + 
                                            u[i, j-1] + u[i, j+1] - 
                                            f_source[i, j])
                    
                    # 2. Apply Weighted Average (Relaxation)
                    u[i, j] = (1 - omega) * u[i, j] + omega * gs_prediction
            
            u_current_norm = np.linalg.norm(u)
            diff = abs(u_current_norm - u_old_norm)
            residuals.append(diff)
            
            if diff < tol:
                print(f"SOR converged at iteration {k}. Time: {time.time()-start_time:.4f}s")
                return u, residuals
                
        return u, residuals