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
        self.fixed_dof = None  # Store fixed DOFs inside the class
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
            
            K_e_ff, _, free_dof, fixed_dof = bc_result  # Extract fixed DOFs
            self.fixed_dof = fixed_dof  # 🔥 Store it for later use

            bc_result = self.frame_solver.apply_boundary_conditions(self.global_geometric_stiffness, np.zeros(self.ndof))
            if bc_result is None or len(bc_result) < 4:
                raise ValueError("Boundary conditions function did not return expected values.")

            K_g_ff, _, _, _ = bc_result
            #print("Condition number of reduced K_e_ff:", np.log10(np.linalg.cond(K_e_ff)))
            #print("Condition number of reduced K_g_ff:", np.log10(np.linalg.cond(K_g_ff)))

            # Debug print statements
            #print(f"Free DOFs: {free_dof}")
            #print(f"Fixed DOFs: {fixed_dof}")  
            #print(f"Total system DOFs: {self.frame_solver.ndof}")
            #print(f"Fixed DOFs: {fixed_dof}")
            #print(f"Free DOFs: {free_dof}")
            #print(f"Expected eigenvector size (should match free DOFs): {len(free_dof)}")
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
            #print(f"Eigenvector shape: {eigenvectors.shape}")
            #print(f"Total expected DOFs (frame_solver.ndof): {self.frame_solver.ndof}")
            #print(f"Eigenvector matrix shape: {eigenvectors.shape}")
            #print(f"Total DOFs in system (before BCs): {self.frame_solver.ndof}")
            #print(f"Free DOFs in system (after BCs): {len(free_dof)}")
            
            # Map eigenvectors back to full DOF system
            # Extract fixed and free DOFs from the existing bc_result
            _, _, free_dof, fixed_dof = bc_result  # Ensure fixed DOFs are correctly extracted

            # Map eigenvectors back to the full system DOFs
            full_mode_shapes = np.zeros((self.frame_solver.ndof, eigenvectors.shape[1]))
            for mode in range(eigenvectors.shape[1]):
                full_mode_shapes[free_dof, mode] = eigenvectors[:, mode]  # Map free DOFs back

            # 🔥 Double-check that fixed DOFs are actually zero
            full_mode_shapes[fixed_dof, :] = 0  # Explicitly set them to zero

            # Debugging print to verify fixed DOFs are zero
            for dof in fixed_dof:
                if not np.allclose(full_mode_shapes[dof, :], 0):
                    #print(f"⚠️ Fixed DOF {dof} is not zero! Setting it again.")
                    full_mode_shapes[dof, :] = 0  # Redundant safety check

            #print(f" Fixed DOFs enforced correctly. Shape of mode shapes: {full_mode_shapes.shape}")

            return eigenvalues, full_mode_shapes

        except Exception as e:
            print(f"Error in solve_eigenvalue_problem: {e}")
            raise

# -----------------------
# Plotting Buckling Mode
# -----------------------    

def hermite_beam_shape_functions(s, L):
    """
    Compute cubic Hermite shape functions for beam bending.

    Parameters
    ----------
    s : float
        Parametric coordinate along the element (0 ≤ s ≤ L).
    L : float
        Element length.

    Returns
    -------
    H1, H2, H3, H4 : float
        Hermite shape function values at `s`.
    """
    xi = s / L  # Normalized coordinate (0 ≤ xi ≤ 1)
    
    H1 = 1 - 3 * xi**2 + 2 * xi**3
    H2 = L * (xi*(1 - xi)**2)
    H3 = 3 * xi**2 - 2 * xi**3
    H4 = L * (xi*(xi**2 - xi))

    return H1, H2, H3, H4

def plot_buckling_mode(frame_solver, mode_shape, scale_factor=1.0, n_points=100):
    """
    Plot the buckled structure using **true Hermite cubic interpolation** for 3D beam bending.

    Parameters
    ----------
    frame_solver : Frame3DSolver
        Instance of Frame3DSolver containing frame geometry and elements.
    mode_shape : np.ndarray
        Eigenvector corresponding to a buckling mode, representing nodal displacements and rotations.
    scale_factor : float, optional
        Factor to scale the mode shape for better visualization (default is 1.0).
    n_points : int, optional
        Number of points per element for interpolation (default is 50).
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract node positions and deformations
    nodes = frame_solver.nodes
    deformed_nodes = {}

    for node, coords in nodes.items():
        idx = frame_solver.node_index_map[node] * 6
        displacement = mode_shape[idx:idx + 3]  # Translational DOFs [u, v, w]
        deformed_nodes[node] = coords + scale_factor * displacement

    # Store all coordinates for axis scaling
    all_x, all_y, all_z = [], [], []

    # Plot undeformed structure
    for elem in frame_solver.elements:
        node1, node2, _ = elem
        coord1 = nodes[node1]
        coord2 = nodes[node2]
        ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]],
                'k--', linewidth=1, label='Undeformed' if elem == frame_solver.elements[0] else "")
        all_x.extend([coord1[0], coord2[0]])
        all_y.extend([coord1[1], coord2[1]])
        all_z.extend([coord1[2], coord2[2]])

    # Plot deformed structure using **true Hermite shape functions**
    for elem in frame_solver.elements:
        node1, node2, _ = elem
        idx1 = frame_solver.node_index_map[node1] * 6
        idx2 = frame_solver.node_index_map[node2] * 6

        # Extract deformed nodal positions
        coord1 = deformed_nodes[node1]
        coord2 = deformed_nodes[node2]

        # Extract displacements and rotations
        u1, v1, w1 = scale_factor * mode_shape[idx1:idx1 + 3]
        u2, v2, w2 = scale_factor * mode_shape[idx2:idx2 + 3]
        theta_x1, theta_y1, theta_z1 = scale_factor * mode_shape[idx1 + 3:idx1 + 6]
        theta_x2, theta_y2, theta_z2 = scale_factor * mode_shape[idx2 + 3:idx2 + 6]

        # Compute element length
        L = np.linalg.norm(coord2 - coord1)
        s_vals = np.linspace(0, L, n_points)

        # Interpolated coordinates using Hermite cubic shape functions
        x_hermite, y_hermite, z_hermite = [], [], []

        for s in s_vals:
            H1, H2, H3, H4 = hermite_beam_shape_functions(s, L)

            # Interpolate displacements
            x = H1 * coord1[0] + H2 * theta_x1 + H3 * coord2[0] + H4 * theta_x2
            y = H1 * coord1[1] + H2 * theta_y1 + H3 * coord2[1] + H4 * theta_y2
            z = H1 * coord1[2] + H2 * theta_z1 + H3 * coord2[2] + H4 * theta_z2

            x_hermite.append(x)
            y_hermite.append(y)
            z_hermite.append(z)

        ax.plot(x_hermite, y_hermite, z_hermite, 'r-', linewidth=2,
                label='Buckled Shape' if elem == frame_solver.elements[0] else "")

        all_x.extend(x_hermite)
        all_y.extend(y_hermite)
        all_z.extend(z_hermite)

    # Set equal axis ranges
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    min_z, max_z = min(all_z), max(all_z)

    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2.0

    mid_x = (max_x + min_x) / 2.0
    mid_y = (max_y + min_y) / 2.0
    mid_z = (max_z + min_z) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Elastic Critical Load Analysis')
    ax.legend()
    plt.show()
