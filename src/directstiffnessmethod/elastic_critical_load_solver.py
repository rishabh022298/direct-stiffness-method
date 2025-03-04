import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .direct_stiffness_method import Frame3DSolver, rotation_matrix_3D, transformation_matrix_3D

# -----------------------
# Computing Local Geometric Stiffness Matrix for 3D Beam Element
# -----------------------
def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
    """
    local element geometric stiffness matrix
    source: p. 258 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        material and geometric parameters:
            L, A, I_rho (polar moment of inertia)
        element forces and moments:
            Fx2, Mx2, My1, Mz1, My2, Mz2
    Context:
        load vector:
            [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]
        DOF vector:
            [u1, v1, w1, th_x1, th_y1, th_z1, u2, v2, w2, th_x2, th_y2, th_z2]
        Equation:
            [load vector] = [stiffness matrix] @ [DOF vector]
    Returns:
        12 x 12 geometric stiffness matrix k_g
    """
    k_g = np.zeros((12, 12))
    # upper triangle off diagonal terms
    k_g[0, 6] = -Fx2 / L
    k_g[1, 3] = My1 / L
    k_g[1, 4] = Mx2 / L
    k_g[1, 5] = Fx2 / 10.0
    k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
    k_g[1, 9] = My2 / L
    k_g[1, 10] = -Mx2 / L
    k_g[1, 11] = Fx2 / 10.0
    k_g[2, 3] = Mz1 / L
    k_g[2, 4] = -Fx2 / 10.0
    k_g[2, 5] = Mx2 / L
    k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
    k_g[2, 9] = Mz2 / L
    k_g[2, 10] = -Fx2 / 10.0
    k_g[2, 11] = -Mx2 / L
    k_g[3, 4] = -1.0 * (2.0 * Mz1 - Mz2) / 6.0
    k_g[3, 5] = (2.0 * My1 - My2) / 6.0
    k_g[3, 7] = -My1 / L
    k_g[3, 8] = -Mz1 / L
    k_g[3, 9] = -Fx2 * I_rho / (A * L)
    k_g[3, 10] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[3, 11] = (My1 + My2) / 6.0
    k_g[4, 7] = -Mx2 / L
    k_g[4, 8] = Fx2 / 10.0
    k_g[4, 9] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[4, 10] = -Fx2 * L / 30.0
    k_g[4, 11] = Mx2 / 2.0
    k_g[5, 7] = -Fx2 / 10.0
    k_g[5, 8] = -Mx2 / L
    k_g[5, 9] = (My1 + My2) / 6.0
    k_g[5, 10] = -Mx2 / 2.0
    k_g[5, 11] = -Fx2 * L / 30.0
    k_g[7, 9] = -My2 / L
    k_g[7, 10] = Mx2 / L
    k_g[7, 11] = -Fx2 / 10.0
    k_g[8, 9] = -Mz2 / L
    k_g[8, 10] = Fx2 / 10.0
    k_g[8, 11] = Mx2 / L
    k_g[9, 10] = (Mz1 - 2.0 * Mz2) / 6.0
    k_g[9, 11] = -1.0 * (My1 - 2.0 * My2) / 6.0
    # add in the symmetric lower triangle
    k_g = k_g + k_g.transpose()
    # add diagonal terms
    k_g[0, 0] = Fx2 / L
    k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
    k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
    k_g[3, 3] = Fx2 * I_rho / (A * L)
    k_g[4, 4] = 2.0 * Fx2 * L / 15.0
    k_g[5, 5] = 2.0 * Fx2 * L / 15.0
    k_g[6, 6] = Fx2 / L
    k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
    k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
    k_g[9, 9] = Fx2 * I_rho / (A * L)
    k_g[10, 10] = 2.0 * Fx2 * L / 15.0
    k_g[11, 11] = 2.0 * Fx2 * L / 15.0
    return k_g

# -----------------------
# Computing Local Stiffness Matrix without Interaction Terms
# -----------------------
def local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(L, A, I_rho, Fx2):
    """
    local element geometric stiffness matrix
    source: p. 257 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        material and geometric parameters:
            L, A, I_rho (polar moment of inertia)
        element forces and moments:
            Fx2
    Context:
        load vector:
            [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]
        DOF vector:
            [u1, v1, w1, th_x1, th_y1, th_z1, u2, v2, w2, th_x2, th_y2, th_z2]
        Equation:
            [load vector] = [stiffness matrix] @ [DOF vector]
    Returns:
        12 x 12 geometric stiffness matrix k_g
    """
    k_g = np.zeros((12, 12))
    # upper triangle off diagonal terms
    k_g[0, 6] = -Fx2 / L
    k_g[1, 5] = Fx2 / 10.0
    k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
    k_g[1, 11] = Fx2 / 10.0
    k_g[2, 4] = -Fx2 / 10.0
    k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
    k_g[2, 10] = -Fx2 / 10.0
    k_g[3, 9] = -Fx2 * I_rho / (A * L)
    k_g[4, 8] = Fx2 / 10.0
    k_g[4, 10] = -Fx2 * L / 30.0
    k_g[5, 7] = -Fx2 / 10
    k_g[5, 11] = -Fx2 * L / 30.0
    k_g[7, 11] = -Fx2 / 10.0
    k_g[8, 10] = Fx2 / 10.0
    # add in the symmetric lower triangle
    k_g = k_g + k_g.transpose()
    # add diagonal terms
    k_g[0, 0] = Fx2 / L
    k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
    k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
    k_g[3, 3] = Fx2 * I_rho / (A * L)
    k_g[4, 4] = 2.0 * Fx2 * L / 15.0
    k_g[5, 5] = 2.0 * Fx2 * L / 15.0
    k_g[6, 6] = Fx2 / L
    k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
    k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
    k_g[9, 9] = Fx2 * I_rho / (A * L)
    k_g[10, 10] = 2.0 * Fx2 * L / 15.0
    k_g[11, 11] = 2.0 * Fx2 * L / 15.0
    return k_g

# -----------------------
# Extracting Moments and Forces for Calculating Stiffness with Interaction terms
# -----------------------
def extract_moments_from_internal_forces(internal_forces, element):
    """
    Extract the axial force and moments from internal forces for a given element.

    Parameters
    ----------
    internal_forces : dict
        Internal forces for each element.
    element : tuple
        Tuple of (node1, node2) defining the element.

    Returns
    -------
    Fx2, Mx2, My1, Mz1, My2, Mz2 : float
        Extracted axial force and moments.
    """
    forces = internal_forces[element]

    # Extract forces and moments from local coordinate system
    Fx2 = forces[6]  # Axial force at node 2
    Mx2 = forces[9]  # Torsional moment at node 2
    My1 = forces[4]  # Bending moment about y at node 1
    Mz1 = forces[5]  # Bending moment about z at node 1
    My2 = forces[10] # Bending moment about y at node 2
    Mz2 = forces[11] # Bending moment about z at node 2

    return Fx2, Mx2, My1, Mz1, My2, Mz2

# -----------------------
# Creating Class for Elastic Critical Load Solver
# -----------------------
class ElasticCriticalLoadSolver:
    """
    Elastic Critical Load Solver for 3D Frames using the Direct Stiffness Method.
    """

    def __init__(self, frame_solver: Frame3DSolver, use_interaction_terms: bool = False):
        """
        Initialize the solver with an instance of Frame3DSolver.
        
        Parameters
        ----------
        frame_solver : Frame3DSolver
            Instance of Frame3DSolver containing frame geometry, 
            material properties, loads, and boundary conditions.
        use_interaction_terms : bool, optional
            If True, uses geometric stiffness with interaction terms.
            If False, uses geometric stiffness without interaction terms.
        """
        self.frame_solver = frame_solver
        self.use_interaction_terms = use_interaction_terms
        self.ndof = frame_solver.ndof
        self.global_geometric_stiffness = np.zeros((self.ndof, self.ndof))

# -----------------------
# Assembling Geometric Stiffness
# -----------------------    
    def assemble_geometric_stiffness(self, d: np.ndarray):
        """
        Assemble the global geometric stiffness matrix using local geometric stiffness matrices.

        Parameters
        ----------
        d : np.ndarray
            Global displacement vector from the static analysis.
        """
        # Compute internal forces first
        internal_forces = self.frame_solver.compute_internal_forces_and_moments(d)

        for elem in self.frame_solver.elements:
            node1, node2, props = elem
            coord1 = self.frame_solver.nodes[node1]
            coord2 = self.frame_solver.nodes[node2]
            L = np.linalg.norm(coord2 - coord1)

            A = props["A"]
            I_rho = props.get("I_rho", props["J"])  # Default value if not provided

            # Extract actual forces and moments
            Fx2, Mx2, My1, Mz1, My2, Mz2 = extract_moments_from_internal_forces(internal_forces, (node1, node2))
            gamma = rotation_matrix_3D(float(coord1[0]), float(coord1[1]), float(coord1[2]),
                                       float(coord2[0]), float(coord2[1]), float(coord2[2]))
            Gamma = transformation_matrix_3D(gamma)            
            # Choose stiffness calculation method
            if self.use_interaction_terms:
                k_geo = local_geometric_stiffness_matrix_3D_beam(
                    L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2
                )
            else:
                k_geo = local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(
                    L, A, I_rho, Fx2
                )
            k_geo_global = Gamma.T @ k_geo @ Gamma
            idx1 = self.frame_solver.node_index_map[node1] * 6
            idx2 = self.frame_solver.node_index_map[node2] * 6
            dof_indices = np.array([idx1, idx1+1, idx1+2, idx1+3, idx1+4, idx1+5,
                                    idx2, idx2+1, idx2+2, idx2+3, idx2+4, idx2+5])

            for i in range(12):
                for j in range(12):
                    self.global_geometric_stiffness[dof_indices[i], dof_indices[j]] += k_geo_global[i, j]

# -----------------------
# Solving Eigenvalue Problem
# -----------------------        
    def solve_eigenvalue_problem(self):
        """
        Solve the elastic critical load problem using:
        (K_e_ff + λ K_g_ff) Δ = 0.

        Returns
        -------
        eigenvalues : np.ndarray
            Array of eigenvalues representing critical load factors.
        eigenvectors : np.ndarray
            Array of eigenvectors representing buckling mode shapes.
        """
        try:
            # Solve for static displacements
            d, _ = self.frame_solver.solve()
            internal_forces = self.frame_solver.compute_internal_forces_and_moments(d)

            # Loop over elements to extract moments and forces
            for elem in self.frame_solver.elements:
                node1, node2 = elem[0], elem[1]  # Extract node indices

                # Extract the internal forces for this element
                Fx2, Mx2, My1, Mz1, My2, Mz2 = extract_moments_from_internal_forces(
                    internal_forces, (node1, node2)
                )

                #print(f"Element {elem}: Fx2={Fx2}, Mx2={Mx2}, My1={My1}, Mz1={Mz1}, My2={My2}, Mz2={Mz2}")
            
            # Assemble geometric stiffness matrix using computed forces/moments
            self.assemble_geometric_stiffness(d)

            # Assemble elastic stiffness matrix
            K_e = self.frame_solver.assemble_stiffness()

            # Apply boundary conditions correctly
            bc_result = self.frame_solver.apply_boundary_conditions(K_e, np.zeros(self.ndof))

            if bc_result is None or len(bc_result) < 4:
                raise ValueError("Boundary conditions function did not return expected values.")

            K_e_ff, _, free_dof, fixed_dof = bc_result  # Ensure `fixed_dof` is extracted

            bc_result = self.frame_solver.apply_boundary_conditions(self.global_geometric_stiffness, np.zeros(self.ndof))
            if bc_result is None or len(bc_result) < 4:
                raise ValueError("Boundary conditions function did not return expected values.")

            K_g_ff, _, _, _ = bc_result
            #print("Condition number of reduced K_e_ff:", np.log10(np.linalg.cond(K_e_ff)))
            #print("Condition number of reduced K_g_ff:", np.log10(np.linalg.cond(K_g_ff)))

            # Debug print statements
            #print(f"Free DOFs: {free_dof}")
            #print(f"Fixed DOFs: {fixed_dof}")  

            # Solve the eigenvalue problem
            eigenvalues, eigenvectors = eig(K_e_ff, -K_g_ff)

            # Convert to real values and remove non-physical eigenvalues
            eigenvalues = np.real(eigenvalues)
            valid_indices = np.where(eigenvalues > 0)[0]
            eigenvalues = eigenvalues[valid_indices]
            eigenvectors = eigenvectors[:, valid_indices]

            # Sort eigenvalues
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            return eigenvalues, eigenvectors

        except Exception as e:
            print(f"Error in solve_eigenvalue_problem: {e}")
            raise

# -----------------------
# Plotting Buckling Mode
# -----------------------    
    def plot_buckling_mode(self, eigenvector, scale=1.0):
        """
        Plot the buckling mode shape corresponding to an eigenvector.
        
        Parameters
        ----------
        eigenvector : np.ndarray
            The eigenvector representing the buckling mode shape.
        scale : float, optional
            Scaling factor for the deformations.
        """
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        for elem in self.frame_solver.elements:
            node1, node2, _ = elem
            
            x = [self.frame_solver.nodes[node1][0], self.frame_solver.nodes[node2][0]]
            y = [self.frame_solver.nodes[node1][1], self.frame_solver.nodes[node2][1]]
            z = [self.frame_solver.nodes[node1][2], self.frame_solver.nodes[node2][2]]
            
            ax.plot(x, y, z, color='blue', label='Undeformed', linewidth=2.5)
            
            idx1 = self.frame_solver.node_index_map[node1] * 6
            idx2 = self.frame_solver.node_index_map[node2] * 6
            
            xd = [x[0] + scale * eigenvector[idx1], x[1] + scale * eigenvector[idx2]]
            yd = [y[0] + scale * eigenvector[idx1+1], y[1] + scale * eigenvector[idx2+1]]
            zd = [z[0] + scale * eigenvector[idx1+2], z[1] + scale * eigenvector[idx2+2]]
            
            ax.plot(xd, yd, zd, color='red', linestyle='--', label='Buckling Mode', linewidth=2.5)
        
        ax.set_title('Buckling Mode Shape', fontsize=20, fontweight='bold')
        ax.set_xlabel('X', fontsize=15, fontweight='bold')
        ax.set_ylabel('Y', fontsize=15, fontweight='bold')
        ax.set_zlabel('Z', fontsize=15, fontweight='bold')
        plt.show()
