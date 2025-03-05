"""
3D Frame Solver using the Direct Stiffness Method

Inputs:
    - Frame geometry: node locations and element connectivity.
    - Element section properties: E, ν, A, Iz, Iy, Iρ, J, local z axis.
      (Note: Iρ is not used in this basic formulation.)
    - Nodal loads: forces and moments applied at given nodes.
    - Boundary conditions: prescribed (supported) displacements/rotations at nodes.

Outputs:
    - Nodal displacements and rotations.
    - Reaction forces and moments at the supports.
    
This script assembles the global stiffness matrix for the structure, applies the
boundary conditions, solves the reduced system for displacements, and then computes
the support reactions.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.linalg import eig
from mpl_toolkits.mplot3d import Axes3D

# -----------------------
# Computing Local Stiffness Matrix
# -----------------------
def local_elastic_stiffness_matrix_3D_beam(E: float, nu: float, A: float, L: float,
                                           Iy: float, Iz: float, J: float) -> np.ndarray:
    """
    Compute the local elastic stiffness matrix for a 3D beam element.
    (Based on McGuire's Matrix Structural Analysis 2nd Edition, p.73)
    """
    # Error Handling
    if not all(isinstance(param, (int, float)) for param in [E, nu, A, L, Iy, Iz, J]):
        raise TypeError("E, nu, A, L, Iy, Iz, and J must be numerical values.")
    if not all(param > 0 for param in [E, A, L, Iy, Iz, J]):
        raise ValueError("E, A, L, Iy, Iz, and J must be positive.")
    if not (0 <= nu < 0.5):
        raise ValueError("Poisson's ratio (nu) must be between 0 and 0.5.")
    
    k_e = np.zeros((12, 12))
    
    # Axial terms - extension along local x axis
    axial_stiffness = E * A / L
    k_e[0, 0] = axial_stiffness
    k_e[0, 6] = -axial_stiffness
    k_e[6, 0] = -axial_stiffness
    k_e[6, 6] = axial_stiffness
    
    # Torsion terms - rotation about local x axis
    torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
    k_e[3, 3] = torsional_stiffness
    k_e[3, 9] = -torsional_stiffness
    k_e[9, 3] = -torsional_stiffness
    k_e[9, 9] = torsional_stiffness
    
    # Bending terms - bending about local z axis
    k_e[1, 1] = E * 12.0 * Iz / L ** 3.0
    k_e[1, 7] = -E * 12.0 * Iz / L ** 3.0
    k_e[7, 1] = -E * 12.0 * Iz / L ** 3.0
    k_e[7, 7] = E * 12.0 * Iz / L ** 3.0
    k_e[1, 5] = E * 6.0 * Iz / L ** 2.0
    k_e[5, 1] = E * 6.0 * Iz / L ** 2.0
    k_e[1, 11] = E * 6.0 * Iz / L ** 2.0
    k_e[11, 1] = E * 6.0 * Iz / L ** 2.0
    k_e[5, 7] = -E * 6.0 * Iz / L ** 2.0
    k_e[7, 5] = -E * 6.0 * Iz / L ** 2.0
    k_e[7, 11] = -E * 6.0 * Iz / L ** 2.0
    k_e[11, 7] = -E * 6.0 * Iz / L ** 2.0
    k_e[5, 5] = E * 4.0 * Iz / L
    k_e[11, 11] = E * 4.0 * Iz / L
    k_e[5, 11] = E * 2.0 * Iz / L
    k_e[11, 5] = E * 2.0 * Iz / L
    
    # Bending terms - bending about local y axis
    k_e[2, 2] = E * 12.0 * Iy / L ** 3.0
    k_e[2, 8] = -E * 12.0 * Iy / L ** 3.0
    k_e[8, 2] = -E * 12.0 * Iy / L ** 3.0
    k_e[8, 8] = E * 12.0 * Iy / L ** 3.0
    k_e[2, 4] = -E * 6.0 * Iy / L ** 2.0
    k_e[4, 2] = -E * 6.0 * Iy / L ** 2.0
    k_e[2, 10] = -E * 6.0 * Iy / L ** 2.0
    k_e[10, 2] = -E * 6.0 * Iy / L ** 2.0
    k_e[4, 8] = E * 6.0 * Iy / L ** 2.0
    k_e[8, 4] = E * 6.0 * Iy / L ** 2.0
    k_e[8, 10] = E * 6.0 * Iy / L ** 2.0
    k_e[10, 8] = E * 6.0 * Iy / L ** 2.0
    k_e[4, 4] = E * 4.0 * Iy / L
    k_e[10, 10] = E * 4.0 * Iy / L
    k_e[4, 10] = E * 2.0 * Iy / L
    k_e[10, 4] = E * 2.0 * Iy / L
    
    return k_e

def check_unit_vector(vec: np.ndarray):
    """
    Verify that the input vector is a unit vector.
    
    Parameters
    ----------
    vec : np.ndarray
        The vector to check.
        
    Raises
    ------
    ValueError
        If the vector is not of unit length.
    """
    # Error Handling
    if not isinstance(vec, np.ndarray):
        raise TypeError("vec must be a numpy array.")
    if vec.shape != (3,):
        raise ValueError("vec must be a 3-element vector.")
        
    if not np.isclose(np.linalg.norm(vec), 1.0):
        raise ValueError("Expected a unit vector for reference vector.")

def check_parallel(vec_1: np.ndarray, vec_2: np.ndarray):
    """
    Check if two vectors are parallel.
    
    Parameters
    ----------
    vec_1 : np.ndarray
        First vector.
    vec_2 : np.ndarray
        Second vector.
        
    Raises
    ------
    ValueError
        If the vectors are parallel.
    """
    # Error Handling
    if not (isinstance(vec_1, np.ndarray) and isinstance(vec_2, np.ndarray)):
        raise TypeError("vec_1 and vec_2 must be numpy arrays.")
    if vec_1.shape != (3,) or vec_2.shape != (3,):
        raise ValueError("vec_1 and vec_2 must be 3-element vectors.")
    
    if np.isclose(np.linalg.norm(np.cross(vec_1, vec_2)), 0.0):
        raise ValueError("Reference vector is parallel to beam axis.")

# -----------------------
# Computing Rotation Matrix
# -----------------------
def rotation_matrix_3D(x1: float, y1: float, z1: float,
                       x2: float, y2: float, z2: float,
                       v_temp: np.ndarray = None) -> np.ndarray:
    """
    Compute the 3D rotation matrix for a beam element.
    (Based on Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition)
    """
    # Error Handling
    if not all(isinstance(coord, (int, float)) for coord in [x1, y1, z1, x2, y2, z2]):
        raise TypeError("Coordinates must be numerical values.")
    
    L = np.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0 + (z2 - z1) ** 2.0)
    if np.isclose(L, 0.0):
        raise ValueError("Element length cannot be zero.")
        
    lxp = (x2 - x1) / L
    mxp = (y2 - y1) / L
    nxp = (z2 - z1) / L
    local_x = np.asarray([lxp, mxp, nxp])

    # Choose a vector to help define the local coordinate system.
    if v_temp is None:
        if np.isclose(lxp, 0.0) and np.isclose(mxp, 0.0):
            v_temp = np.array([0.0, 1.0, 0.0])
        else:
            v_temp = np.array([0.0, 0.0, 1.0])
    else:
        check_unit_vector(v_temp)
        check_parallel(local_x, v_temp)
    
    # Compute the local y axis.
    local_y = np.cross(v_temp, local_x)
    local_y = local_y / np.linalg.norm(local_y)

    # Compute the local z axis.
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)

    # Assemble the rotation matrix.
    gamma = np.vstack((local_x, local_y, local_z))
    
    # Error Handling
    if gamma.shape != (3, 3):
        raise ValueError("Rotation matrix should be 3x3.")
    
    return gamma

# -----------------------
# Computing Transformation Matrix in 3D
# -----------------------
def transformation_matrix_3D(gamma: np.ndarray) -> np.ndarray:
    """
    Compute the 12x12 transformation matrix for a 3D beam element.
    (Based on Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition)
    """
    # Error Handling
    if not isinstance(gamma, np.ndarray):
        raise TypeError("gamma must be a numpy array.")
    if gamma.shape != (3, 3):
        raise ValueError("gamma must be a 3x3 rotation matrix.")
    
    Gamma = np.zeros((12, 12))
    Gamma[0:3, 0:3] = gamma
    Gamma[3:6, 3:6] = gamma
    Gamma[6:9, 6:9] = gamma
    Gamma[9:12, 9:12] = gamma
    
    return Gamma

# -----------------------
# Creating Class for the Solver
# -----------------------
class Frame3DSolver:
    """
    A solver for 3D frame analysis using the Direct Stiffness Method.
    """
    
    def __init__(self, nodes: dict, elements: list, loads: dict, supports: dict):
        # Error Handling
        if not isinstance(nodes, dict):
            raise TypeError("nodes must be a dictionary.")
        if not isinstance(elements, list):
            raise TypeError("elements must be a list.")
        if not isinstance(loads, dict):
            raise TypeError("loads must be a dictionary.")
        if not isinstance(supports, dict):
            raise TypeError("supports must be a dictionary.")
        
        self.nodes = nodes
        self.elements = elements
        self.loads = loads
        self.supports = supports

        self.node_ids = sorted(nodes.keys())
        self.n_nodes = len(self.node_ids)
        self.ndof = self.n_nodes * 6

        self.node_index_map = {node_id: i for i, node_id in enumerate(self.node_ids)}
    
# -----------------------
# Assembling Global Stiffness Matrix
# -----------------------
    def assemble_stiffness(self) -> np.ndarray:
        """
        Assemble and return the global stiffness matrix.
        """
        K = np.zeros((self.ndof, self.ndof))
        
        for elem in self.elements:
            if len(elem) != 3:
                raise ValueError("Each element must be a tuple of (node1, node2, properties).")
            
            node1, node2, props = elem
            coord1 = self.nodes[node1]
            coord2 = self.nodes[node2]
            
            L = np.linalg.norm(coord2 - coord1)
            if np.isclose(L, 0.0):
                raise ValueError("Element length cannot be zero.")
            
            E = props["E"]
            nu = props["nu"]
            A = props["A"]
            Iz = props["Iz"]
            Iy = props["Iy"]
            J = props["J"]
            local_z = props.get("local_z", None)
            
            gamma = rotation_matrix_3D(float(coord1[0]), float(coord1[1]), float(coord1[2]),
                                       float(coord2[0]), float(coord2[1]), float(coord2[2]),
                                       v_temp=local_z)
            Gamma = transformation_matrix_3D(gamma)
            k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
            k_global = Gamma.T @ k_local @ Gamma
            
            idx1 = self.node_index_map[node1] * 6
            idx2 = self.node_index_map[node2] * 6
            dof_indices = np.array([idx1, idx1+1, idx1+2, idx1+3, idx1+4, idx1+5,
                                    idx2, idx2+1, idx2+2, idx2+3, idx2+4, idx2+5])
            
            for i in range(12):
                for j in range(12):
                    K[dof_indices[i], dof_indices[j]] += k_global[i, j]
                    
        return K

# -----------------------
# Assembling Load Vector
# -----------------------
    def assemble_load_vector(self) -> np.ndarray:
        """
        Assemble and return the global load vector.
        
        Returns
        -------
        F : np.ndarray
            Global load vector of size (ndof,).
        """
        # Error Handling
        if not isinstance(self.loads, dict):
            raise TypeError("loads must be a dictionary.")
        
        F = np.zeros(self.ndof)
        for node_id, load in self.loads.items():
            if not isinstance(load, np.ndarray):
                raise TypeError(f"Load at node {node_id} must be a numpy array.")
            if load.shape != (6,):
                raise ValueError(f"Load vector at node {node_id} must be of length 6.")
            
            idx = self.node_index_map[node_id] * 6
            F[idx:idx+6] += load
        
        return F

# -----------------------
# Applying Boundary Conditions
# -----------------------
    def apply_boundary_conditions(self, K: np.ndarray, F: np.ndarray):
        """
        Apply boundary conditions by partitioning the global system into free and fixed DOFs.
        
        Parameters
        ----------
        K : np.ndarray
            The global stiffness matrix.
        F : np.ndarray
            The global load vector.
            
        Returns
        -------
        K_reduced : np.ndarray
            Reduced stiffness matrix corresponding to free DOFs.
        F_reduced : np.ndarray
            Reduced load vector corresponding to free DOFs.
        free_dof : np.ndarray
            Array of indices corresponding to free DOFs.
        fixed_dof : np.ndarray
            Array of indices corresponding to fixed DOFs.
        """
        # Error Handling
        if not isinstance(K, np.ndarray) or not isinstance(F, np.ndarray):
            raise TypeError("K and F must be numpy arrays.")
        if K.shape != (self.ndof, self.ndof):
            raise ValueError("K must be a square matrix of size (ndof, ndof).")
        if F.shape != (self.ndof,):
            raise ValueError("F must be a vector of length ndof.")
        
        fixed_dof = []
        
        # Loop over nodes and collect DOFs that are constrained.
        for node_id, bc in self.supports.items():
            if not isinstance(bc, list) or len(bc) != 6:
                raise ValueError(f"Boundary condition at node {node_id} must be a list of length 6.")
            if not all(isinstance(x, bool) for x in bc):
                raise TypeError(f"Boundary conditions at node {node_id} must be booleans.")
            
            idx = self.node_index_map[node_id] * 6
            for i, is_fixed in enumerate(bc):
                if is_fixed:
                    fixed_dof.append(idx + i)
        
        fixed_dof = np.array(fixed_dof)
        all_dof = np.arange(self.ndof)
        free_dof = np.setdiff1d(all_dof, fixed_dof)
        
        # Reduce the system.
        K_reduced = K[np.ix_(free_dof, free_dof)]
        F_reduced = F[free_dof]
        
        return K_reduced, F_reduced, free_dof, fixed_dof
# -----------------------
# Computing Required Quantities
# -----------------------
    def solve(self):
        """
        Assemble the system, apply boundary conditions, and solve for nodal displacements.
        Also computes the reaction forces at the supports.
        
        Returns
        -------
        d : np.ndarray
            Full displacement vector (size ndof) with nodal displacements/rotations.
        reactions : np.ndarray
            Global reaction vector computed as (K*d - F).
        """
        # Assemble global stiffness matrix and load vector.
        K = self.assemble_stiffness()
        F = self.assemble_load_vector()
        
        # Apply boundary conditions.
        K_reduced, F_reduced, free_dof, fixed_dof = self.apply_boundary_conditions(K, F)
        
        # Check for numerical singularity using condition number
        cond_number = np.linalg.cond(K_reduced)
        if cond_number > 1e12:  # Threshold for numerical singularity
            raise np.linalg.LinAlgError(f"Global stiffness matrix is nearly singular (Condition Number: {cond_number})")

        # Solve for displacements at free DOFs.
        d = np.zeros(self.ndof)
        try:
            d_free = np.linalg.solve(K_reduced, F_reduced)
            
            # Check if solution has NaNs or Infs
            if np.any(np.isnan(d_free)) or np.any(np.isinf(d_free)):
                raise np.linalg.LinAlgError("Solution contains NaN or Inf, indicating singular system.")
            
            # Check for unreasonably large values in the solution vector
            if np.any(np.abs(d_free) > 1e6):  # Only check for unreasonably large values
                raise np.linalg.LinAlgError("Unreasonably large displacements, indicating numerical instability.")
            
            d[free_dof] = d_free

        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Error solving system of equations: {e}")
        
        # Compute reaction forces.
        reactions = K @ d - F
        
        # Error Handling for Results
        if reactions.shape != (self.ndof,):
            raise ValueError("Reactions vector must be of length ndof.")
        
        return d, reactions

# -----------------------
# Compute Internal Forces and Moments
# -----------------------
    def compute_internal_forces_and_moments(self, d: np.ndarray):
        """
        Compute internal forces and moments for each member in local coordinates.
        
        Parameters
        ----------
        d : np.ndarray
            Global displacement vector.
            
        Returns
        -------
        internal_forces : dict
            Internal forces and moments in local coordinates for each element.
        """
        internal_forces = {}
        
        for elem in self.elements:
            node1, node2, props = elem
            coord1 = self.nodes[node1]
            coord2 = self.nodes[node2]
            
            L = np.linalg.norm(coord2 - coord1)
            
            E = props["E"]
            nu = props["nu"]
            A = props["A"]
            Iz = props["Iz"]
            Iy = props["Iy"]
            J = props["J"]
            local_z = props.get("local_z", None)
            
            gamma = rotation_matrix_3D(float(coord1[0]), float(coord1[1]), float(coord1[2]),
                                    float(coord2[0]), float(coord2[1]), float(coord2[2]),
                                    v_temp=local_z)
            #print(f"Rotation matrix (gamma) for element ({node1}, {node2}):\n{gamma}\n")
            Gamma = transformation_matrix_3D(gamma)
            k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
            
            idx1 = self.node_index_map[node1] * 6
            idx2 = self.node_index_map[node2] * 6
            dof_indices = np.array([idx1, idx1+1, idx1+2, idx1+3, idx1+4, idx1+5,
                                    idx2, idx2+1, idx2+2, idx2+3, idx2+4, idx2+5])
            
            d_elem = d[dof_indices]
            d_local = Gamma @ d_elem
            
            # Use a tuple of node identifiers as the key (hashable)
            internal_forces[(node1, node2)] = k_local @ d_local
            f_global = Gamma.T @ internal_forces[(node1, node2)]
        return internal_forces

# -----------------------
# Plot Internal Forces and Moments
# -----------------------
    def plot_internal_forces_and_moments(self, internal_forces: dict):
        """
        Plot internal forces and moments for each member in local coordinates.
        
        Parameters
        ----------
        internal_forces : dict
            Internal forces and moments in local coordinates for each element.
        """
        for elem, forces in internal_forces.items():
            # Unpack the tuple (node1, node2)
            node1, node2 = elem
            
            x = [0, np.linalg.norm(self.nodes[node2] - self.nodes[node1])]
            
            fig, ax = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle(f'Internal Forces and Moments for Element {elem}', fontsize=25, fontweight='bold')
            
            # Axial Force
            ax[0, 0].plot(x, forces[[0, 6]], label='$F_x$ (Axial Force)', linewidth=2.5)
            ax[0, 0].set_title('$F_x$ (Axial Force)', fontsize=25, fontweight='bold')
            ax[0, 0].legend(fontsize=20)
            ax[0, 0].grid(True)

            # Shear Force in Local y Direction
            ax[0, 1].plot(x, forces[[1, 7]], label='$F_y$ (Shear Force)', linewidth=2.5)
            ax[0, 1].set_title('$F_y$ (Shear Force)', fontsize=25, fontweight='bold')
            ax[0, 1].legend(fontsize=20)
            ax[0, 1].grid(True)

            # Shear Force in Local z Direction
            ax[1, 0].plot(x, forces[[2, 8]], label='$F_z$ (Shear Force)', linewidth=2.5)
            ax[1, 0].set_title('$F_z$ (Shear Force)', fontsize=25, fontweight='bold')
            ax[1, 0].legend(fontsize=20)
            ax[1, 0].grid(True)

            # Bending Moment about Local z Axis
            ax[1, 1].plot(x, forces[[5, 11]], label='$M_z$ (Bending Moment)', linewidth=2.5)
            ax[1, 1].set_title('$M_z$ (Bending Moment)', fontsize=25, fontweight='bold')
            ax[1, 1].legend(fontsize=20)
            ax[1, 1].grid(True)

            plt.show()

# -----------------------
# Plot Deformed Shape
# -----------------------
    def plot_deformed_shape(self, d: np.ndarray, scale: float = 1.0):
        """
        Plot the undeformed and deformed shape of the whole structure.
        
        Parameters
        ----------
        d : np.ndarray
            Global displacement vector.
        scale : float, optional
            Scaling factor for the deformations.
        """
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        for elem in self.elements:
            node1, node2, _ = elem
            
            x = [self.nodes[node1][0], self.nodes[node2][0]]
            y = [self.nodes[node1][1], self.nodes[node2][1]]
            z = [self.nodes[node1][2], self.nodes[node2][2]]
            
            ax.plot(x, y, z, color='blue', linestyle='--', label='Undeformed', linewidth=2.5)
            
            idx1 = self.node_index_map[node1] * 6
            idx2 = self.node_index_map[node2] * 6
            
            xd = [x[0] + scale * d[idx1], x[1] + scale * d[idx2]]
            yd = [y[0] + scale * d[idx1+1], y[1] + scale * d[idx2+1]]
            zd = [z[0] + scale * d[idx1+2], z[1] + scale * d[idx2+2]]
            
            ax.plot(xd, yd, zd, color='red', label='Deformed', linewidth=2.5)
        
        ax.set_title('Deformed Shape of the Structure', fontsize=20, fontweight='bold')
        ax.set_xlabel('X', fontsize=15, fontweight='bold')
        ax.set_ylabel('Y', fontsize=15, fontweight='bold')
        ax.set_zlabel('Z', fontsize=15, fontweight='bold')
        ax.legend()
        plt.show()
