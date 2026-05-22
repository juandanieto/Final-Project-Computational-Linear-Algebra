import numpy as np
import matplotlib.pyplot as plt
from solvers import IterativeSolvers  # Imports the class from the previous file

# ==========================================
# PROBLEM CONFIGURATION (Physics Model)
# ==========================================
def setup_problem(N):
    """
    Creates the grid and the source term (Heat Source).
    N: Grid size (NxN)
    """
    h = 1.0 / (N - 1)
    u_init = np.zeros((N, N))
    
    # 'f' represents the heat source/sink
    f = np.zeros((N, N))
    
    # Heat source in the center
    mid = N // 2
    f[mid-2:mid+2, mid-2:mid+2] = 5000.0 
    
    # Heat sink (cooling area)
    f[5:10, 5:10] = -2000.0
    
    # Scale f by h^2 (as per the discretized Poisson equation)
    # This avoids doing the multiplication inside the loops
    rhs = f * h**2
    
    return u_init, rhs

# ==========================================
# STUDY 1: JACOBI - Mesh Size Sensitivity
# ==========================================
def experiment_jacobi_mesh_size():
    print("\n=== Running Experiment 1: Jacobi vs Mesh Size ===")
    mesh_sizes = [20, 40, 60]
    tol = 1e-5
    
    plt.figure(figsize=(8, 6))
    
    for N in mesh_sizes:
        print(f"Testing Mesh Size: {N}x{N}")
        u_init, f = setup_problem(N)
        _, residuals = IterativeSolvers.jacobi(u_init, f, tol=tol, max_iter=4000)
        
        plt.semilogy(residuals, label=f'Mesh {N}x{N}', linewidth=2)

    plt.title("Study 1: Jacobi Convergence degradation with Mesh Size")
    plt.xlabel("Iteration Number (k)")
    plt.ylabel("Residual Error Norm (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig("Fig1_Jacobi_Mesh_Sensitivity.png")
    print("Saved plot: Fig1_Jacobi_Mesh_Sensitivity.png")

# ==========================================
# STUDY 2: GAUSS-SEIDEL - Error Smoothing Visualization
# ==========================================
def experiment_gs_error_evolution():
    print("\n=== Running Experiment 2: Gauss-Seidel Error Evolution ===")
    N = 40
    u_init, f = setup_problem(N)
    
    # 1. Get a "Reference Solution" using SOR with high precision
    print("Computing reference solution (High Precision)...")
    w_opt = 2 / (1 + np.sin(np.pi/N))
    u_exact, _ = IterativeSolvers.sor(u_init, f, w_opt, tol=1e-12, max_iter=5000)
    
    # 2. Capture snapshots of Gauss-Seidel at specific iterations
    iterations_to_capture = [10, 50, 200]
    snapshots = []
    
    u = u_init.copy()
    current_k = 0
    
    # Manual loop to control snapshots
    for target_k in iterations_to_capture:
        steps = target_k - current_k
        
        # Run GS logic for 'steps' amount
        for _ in range(steps):
            for i in range(1, N-1):
                for j in range(1, N-1):
                    u[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - f[i, j])
        
        current_k = target_k
        
        # Calculate Absolute Error Matrix compared to exact solution
        error_matrix = np.abs(u - u_exact)
        snapshots.append((target_k, error_matrix))

    # 3. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (iters, err_mat) in zip(axes, snapshots):
        im = ax.imshow(err_mat, cmap='hot', origin='lower')
        ax.set_title(f"Absolute Error (Iter {iters})")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle("Study 2: Error Smoothing in Gauss-Seidel Method", fontsize=14)
    plt.savefig("Fig2_GS_Error_Evolution.png")
    print("Saved plot: Fig2_GS_Error_Evolution.png")

# ==========================================
# STUDY 3: SOR - Omega Parameter Impact
# ==========================================
def experiment_sor_omega():
    print("\n=== Running Experiment 3: SOR Omega Sensitivity ===")
    N = 50
    u_init, f = setup_problem(N)
    tol = 1e-6
    
    # Theoretical Optimal Omega for this problem
    w_opt_theoretical = 2 / (1 + np.sin(np.pi/N))
    
    omegas = [1.0, 1.5, w_opt_theoretical, 1.95]
    labels = ["1.0 (Gauss-Seidel)", "1.5 (Under-relaxed)", f"{w_opt_theoretical:.3f} (Optimal)", "1.95 (High risk)"]
    colors = ['blue', 'orange', 'green', 'red']
    
    plt.figure(figsize=(9, 6))
    
    for w, lbl, col in zip(omegas, labels, colors):
        _, residuals = IterativeSolvers.sor(u_init, f, w, tol=tol, max_iter=2000)
        plt.semilogy(residuals, label=f'Omega = {lbl}', color=col, linewidth=2)
        
    plt.title(f"Study 3: Impact of Relaxation Parameter (Grid {N}x{N})")
    plt.xlabel("Iteration Number")
    plt.ylabel("Residual Error (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig("Fig3_SOR_Omega_Impact.png")
    print("Saved plot: Fig3_SOR_Omega_Impact.png")

if __name__ == "__main__":
    # Run all experiments
    experiment_jacobi_mesh_size()
    experiment_gs_error_evolution()
    experiment_sor_omega()
    print("\nAll experiments completed successfully.")