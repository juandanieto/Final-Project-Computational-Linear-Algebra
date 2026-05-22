## Project Overview
This project implements and analyzes iterative numerical methods to solve the 2D Poisson equation with customized heat sources and sinks. 

### Key Studies Included:
* **Study 1 (Jacobi Method):** Analyzes convergence degradation and scalability constraints across varying mesh sizes ($20\times20$, $40\times40$, and $60\times60$).
* **Study 2 (Gauss-Seidel Method):** Captures high-frequency error-smoothing snapshots ($10$, $50$, and $200$ iterations) mapped against a high-precision reference solution.
* **Study 3 (SOR Method):** Evaluates relaxation parameter ($\omega$) sensitivity, verifying empirical performance against the theoretical optimal relaxation factor ($\omega_{opt}$).
