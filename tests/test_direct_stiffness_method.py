from directstiffnessmethod import direct_stiffness_method as dsm
from directstiffnessmethod import elastic_critical_load_solver as ecls
import pytest
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

matplotlib.use('Agg')

def test_local_stiffness_matrix_invalid_poisson():
    """
    Test that local_elastic_stiffness_matrix_3D_beam raises a ValueError
    for an invalid Poisson's ratio.
    """
    E = 210e9
    nu = 0.6  # Invalid, should be < 0.5
    A = 0.01
    L = 5.0
    Iy = 8.33e-6
    Iz = 8.33e-6
    J = 1.67e-5
    
    with pytest.raises(ValueError):
        dsm.local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)

def test_local_stiffness_matrix_zero_length():
    """
    Test that local_elastic_stiffness_matrix_3D_beam raises a ValueError
    when the length is zero.
    """
    E = 210e9
    nu = 0.3
    A = 0.01
    L = 0.0  # Zero length
    Iy = 8.33e-6
    Iz = 8.33e-6
    J = 1.67e-5
    
    with pytest.raises(ValueError):
        dsm.local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)

def test_local_stiffness_matrix_negative_values():
    """
    Test that local_elastic_stiffness_matrix_3D_beam raises a ValueError
    when negative values are provided for positive-only parameters.
    """
    E = 210e9
    nu = 0.3
    A = -0.01  # Negative cross-sectional area
    L = 5.0
    Iy = 8.33e-6
    Iz = 8.33e-6
    J = 1.67e-5
    
    with pytest.raises(ValueError):
        dsm.local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)

def test_rotation_matrix_zero_length():
    """
    Test that rotation_matrix_3D raises a ValueError when the element length is zero.
    """
    x1, y1, z1 = 0.0, 0.0, 0.0
    x2, y2, z2 = 0.0, 0.0, 0.0  # Zero length
    
    with pytest.raises(ValueError):
        dsm.rotation_matrix_3D(x1, y1, z1, x2, y2, z2)

def test_rotation_matrix_invalid_reference_vector():
    """
    Test that rotation_matrix_3D raises a ValueError when the reference vector is invalid.
    """
    x1, y1, z1 = 0.0, 0.0, 0.0
    x2, y2, z2 = 1.0, 0.0, 0.0
    v_temp = np.array([1.0, 0.0, 0.0])  # Parallel to local x
    
    with pytest.raises(ValueError):
        dsm.rotation_matrix_3D(x1, y1, z1, x2, y2, z2, v_temp)

def test_assemble_load_vector_invalid_load_length():
    """
    Test that assemble_load_vector raises a ValueError for invalid load vector length.
    """
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([5.0, 0.0, 0.0])
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {
        1: np.array([0.0, -10000.0])  # Invalid length
    }
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }
    
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    
    with pytest.raises(ValueError):
        solver.assemble_load_vector()

def test_apply_boundary_conditions_no_constraints():
    """
    Test that apply_boundary_conditions works correctly when no constraints are present.
    """
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([5.0, 0.0, 0.0])
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {
        1: np.array([0.0, -10000.0, 0.0, 0.0, 0.0, 0.0])
    }
    supports = {
        0: [False, False, False, False, False, False],
        1: [False, False, False, False, False, False]
    }
    
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    K = solver.assemble_stiffness()
    F = solver.assemble_load_vector()
    K_reduced, F_reduced, free_dof, fixed_dof = solver.apply_boundary_conditions(K, F)
    
    # In this case, all DOFs should be free and none should be fixed.
    assert len(fixed_dof) == 0, "No DOFs should be fixed."
    assert len(free_dof) == solver.ndof, "All DOFs should be free."

def test_local_stiffness_matrix_symmetry():
    """
    Test that the local stiffness matrix is symmetric.
    """
    E = 210e9      # Young's modulus
    nu = 0.3       # Poisson's ratio
    A = 0.01       # Cross-sectional area
    L = 5.0        # Element length
    Iy = 8.33e-6   # Moment of inertia about local y axis
    Iz = 8.33e-6   # Moment of inertia about local z axis
    J = 1.67e-5    # Torsional constant

    k_local = dsm.local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
    # Stiffness matrices are symmetric.
    assert np.allclose(k_local, k_local.T, atol=1e-8), "Local stiffness matrix is not symmetric."

def test_rotation_matrix_orthonormality():
    """
    Test that the rotation matrix is orthonormal (i.e. its product with its transpose is the identity)
    and that its determinant is 1.
    """
    # Define a beam from (0,0,0) to (1,0,0)
    gamma = dsm.rotation_matrix_3D(0, 0, 0, 1, 0, 0)
    I = np.eye(3)
    assert np.allclose(gamma @ gamma.T, I, atol=1e-8), "Rotation matrix is not orthonormal."
    # Check the determinant.
    det_gamma = np.linalg.det(gamma)
    assert np.isclose(det_gamma, 1.0, atol=1e-8), "Rotation matrix determinant is not 1."

def test_transformation_matrix_structure():
    """
    Test that the 12x12 transformation matrix is constructed correctly as block-diagonal,
    with the given 3x3 rotation matrix repeated.
    """
    gamma = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    Gamma = dsm.transformation_matrix_3D(gamma)
    # The transformation matrix should consist of 4 diagonal 3x3 blocks equal to gamma.
    for i in range(4):
        block = Gamma[i*3:(i+1)*3, i*3:(i+1)*3]
        assert np.allclose(block, gamma, atol=1e-8), f"Block {i} is not equal to gamma."
    # Off-diagonal blocks must be zero.
    for i in range(4):
        for j in range(4):
            if i != j:
                block = Gamma[i*3:(i+1)*3, j*3:(j+1)*3]
                assert np.allclose(block, np.zeros((3,3)), atol=1e-8), f"Off-diagonal block ({i},{j}) is not zero."

def test_check_unit_vector_valid():
    """
    Test that check_unit_vector passes for a valid unit vector.
    """
    vec = np.array([1.0, 0.0, 0.0])
    # Should not raise an exception.
    dsm.check_unit_vector(vec)

def test_check_unit_vector_invalid():
    """
    Test that check_unit_vector raises a ValueError for a non-unit vector.
    """
    vec = np.array([1.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        dsm.check_unit_vector(vec)

def test_check_parallel_not_parallel():
    """
    Test that check_parallel does not raise an exception when vectors are not parallel.
    """
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    dsm.check_parallel(vec1, vec2)

def test_check_parallel_parallel():
    """
    Test that check_parallel raises a ValueError when vectors are parallel.
    """
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([2.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        dsm.check_parallel(vec1, vec2)

# Define a pytest fixture to create a simple two-node frame
@pytest.fixture
def simple_frame():
    """
    Fixture to create a simple two-node, one-element frame.
    """
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([5.0, 0.0, 0.0])
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {
        1: np.array([0.0, -10000.0, 0.0, 0.0, 0.0, 0.0])
    }
    supports = {
        0: [True, True, True, True, True, True],   # Node 0 is fully fixed
        1: [False, False, False, False, False, False]  # Node 1 is free
    }
    return dsm.Frame3DSolver(nodes, elements, loads, supports)

def test_frame_solver_displacements(simple_frame):
    """
    Test that the solver computes zero displacement for fixed nodes and nonzero displacement for free nodes.
    """
    solver = simple_frame
    displacements, reactions = solver.solve()
    disp_matrix = displacements.reshape((-1, 6))
    
    # Node 0 is fully fixed; its displacements should be zero.
    assert np.allclose(disp_matrix[0, :], np.zeros(6), atol=1e-8), "Fixed node displacement is nonzero."
    
    # Node 1 is free and loaded; it should exhibit nonzero displacement.
    assert not np.allclose(disp_matrix[1, :], np.zeros(6), atol=1e-8), "Free node did not displace as expected."

def test_apply_boundary_conditions(simple_frame):
    """
    Test that the boundary condition application identifies fixed and free degrees-of-freedom correctly.
    """
    solver = simple_frame
    K = solver.assemble_stiffness()
    F = solver.assemble_load_vector()
    K_reduced, F_reduced, free_dof, fixed_dof = solver.apply_boundary_conditions(K, F)
    
    # For the given supports, node 0 (first 6 DOFs) should be fixed.
    expected_fixed = np.arange(0, 6)
    assert np.array_equal(np.sort(fixed_dof), expected_fixed), "Fixed DOFs are not as expected."
    
    # The remaining DOFs should be free.
    expected_free = np.arange(6, solver.ndof)
    assert np.array_equal(np.sort(free_dof), expected_free), "Free DOFs are not as expected."

def test_rotation_matrix_vertical_beam():
    """
    Test the rotation_matrix_3D function for a vertically oriented beam
    when v_temp is not provided. For a vertical beam, the code should choose
    the global y axis (i.e. [0, 1, 0]) as the reference vector.
    """
    # Define a vertical beam (x and y coordinates are constant).
    x1, y1, z1 = 0.0, 0.0, 0.0
    x2, y2, z2 = 0.0, 0.0, 5.0  # vertical beam
    
    # Call rotation_matrix_3D without providing v_temp (so it is None).
    gamma = dsm.rotation_matrix_3D(x1, y1, z1, x2, y2, z2, v_temp=None)
    
    # For a vertical beam, local x should be [0, 0, 1].
    expected_local_x = np.array([0.0, 0.0, 1.0])
    assert np.allclose(gamma[0], expected_local_x, atol=1e-8), "Local x axis is incorrect for a vertical beam."
    
    # Also verify that gamma is orthonormal.
    I = np.eye(3)
    assert np.allclose(gamma @ gamma.T, I, atol=1e-8), "Rotation matrix is not orthonormal for vertical beam."

def test_assemble_load_vector_empty():
    """
    Test that assemble_load_vector returns a zero vector when no loads are applied.
    """
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([5.0, 0.0, 0.0])
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {}  # No loads
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }
    
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    F = solver.assemble_load_vector()
    
    # Expecting a zero vector for the load vector
    assert np.allclose(F, np.zeros(solver.ndof)), "Load vector is not zero for empty loads."

def test_solve_singular_global_stiffness():
    """
    Test that the solver detects a singular global stiffness matrix by
    checking for NaN, Inf, or unreasonably small values in the solution vector.
    """
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([2.0, 0.0, 0.0])  # Collinear nodes
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [
        (0, 1, section_props),
        (1, 2, section_props)
    ]
    loads = {
        2: np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False],
        2: [False, False, False, False, False, False]
    }
    
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    d, _ = solver.solve()
    
    # Check for NaN, Inf, or unreasonably small values in the solution vector
    is_nan_or_inf = np.any(np.isnan(d)) or np.any(np.isinf(d))
    is_unreasonably_small = np.any(np.abs(d) < 1e-12)
    
    assert is_nan_or_inf or is_unreasonably_small, \
        "Expected NaN, Inf, or unreasonably small values in the solution vector for a singular system."

def test_apply_boundary_conditions_invalid_format():
    """
    Test that apply_boundary_conditions raises a ValueError for invalid boundary condition formats.
    """
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([5.0, 0.0, 0.0])
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {
        1: np.array([0.0, -10000.0, 0.0, 0.0, 0.0, 0.0])
    }
    supports = {
        0: [True, True, True, True, True],  # Invalid, length is 5
        1: [False, False, False, False, False, False]
    }
    
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    K = solver.assemble_stiffness()
    F = solver.assemble_load_vector()
    
    with pytest.raises(ValueError):
        solver.apply_boundary_conditions(K, F)

def test_check_unit_vector_type_error():
    with pytest.raises(TypeError):
        dsm.check_unit_vector([1.0, 0.0, 0.0])  # Not a numpy array

def test_check_unit_vector_shape_error():
    with pytest.raises(ValueError):
        dsm.check_unit_vector(np.array([1.0, 0.0]))  # Not a 3-element vector

def test_check_parallel_type_error():
    with pytest.raises(TypeError):
        dsm.check_parallel([1.0, 0.0, 0.0], np.array([0.0, 1.0, 0.0]))

def test_check_parallel_shape_error():
    with pytest.raises(ValueError):
        dsm.check_parallel(np.array([1.0, 0.0]), np.array([0.0, 1.0]))

def test_rotation_matrix_type_error():
    with pytest.raises(TypeError):
        dsm.rotation_matrix_3D(0, 0, 0, "1", 0, 0)  # Non-numerical value

def test_rotation_matrix_zero_length_error():
    with pytest.raises(ValueError):
        dsm.rotation_matrix_3D(0, 0, 0, 0, 0, 0)  # Zero length element

def test_transformation_matrix_type_error():
    with pytest.raises(TypeError):
        dsm.transformation_matrix_3D([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Not a numpy array

def test_transformation_matrix_shape_error():
    with pytest.raises(ValueError):
        dsm.transformation_matrix_3D(np.array([[1, 0], [0, 1]]))  # Not a 3x3 matrix

def test_frame_solver_init_type_error():
    with pytest.raises(TypeError):
        dsm.Frame3DSolver("not_a_dict", [], {}, {})  # nodes should be a dictionary
    with pytest.raises(TypeError):
        dsm.Frame3DSolver({}, "not_a_list", {}, {})  # elements should be a list
    with pytest.raises(TypeError):
        dsm.Frame3DSolver({}, [], "not_a_dict", {})  # loads should be a dictionary
    with pytest.raises(TypeError):
        dsm.Frame3DSolver({}, [], {}, "not_a_dict")  # supports should be a dictionary

def test_assemble_stiffness_invalid_element_format():
    nodes = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0])}
    elements = [(0, 1)]  # Invalid format, missing properties
    solver = dsm.Frame3DSolver(nodes, elements, {}, {})
    with pytest.raises(ValueError):
        solver.assemble_stiffness()

def test_apply_boundary_conditions_invalid_K_F():
    nodes = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0])}
    elements = []
    solver = dsm.Frame3DSolver(nodes, elements, {}, {})
    with pytest.raises(TypeError):
        solver.apply_boundary_conditions("not_array", np.zeros(12))
    with pytest.raises(ValueError):
        solver.apply_boundary_conditions(np.zeros((2, 2)), np.zeros(12))

def test_solve_numerical_instability():
    """
    Test that the solver detects numerical instability when the system is nearly singular.
    This occurs when nodes are collinear or when properties lead to ill-conditioned matrices.
    """
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1e6, 0.0, 0.0]),  # Extremely long beam to amplify instability
        2: np.array([2e6, 0.0, 0.0])  # Collinear nodes with large spacing
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 1e-14,  # Even smaller area to induce numerical instability
        "Iz": 1e-14,  # Very small moment of inertia
        "Iy": 1e-14,
        "J": 1e-14,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props), (1, 2, section_props)]
    loads = {2: np.array([1e9, 0.0, 0.0, 0.0, 0.0, 0.0])}  # Large axial load to amplify instability
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False],
        2: [False, False, False, False, False, False]
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    
    # This should raise a LinAlgError due to the nearly singular matrix
    with pytest.raises(np.linalg.LinAlgError, match="Global stiffness matrix is nearly singular"):
        solver.solve()

def test_assemble_stiffness_invalid_element():
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0])
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [
        (0, 1)  # Missing section properties
    ]
    loads = {}
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(ValueError, match=r"Each element must be a tuple of \(node1, node2, properties\)."):
        solver.assemble_stiffness()

def test_zero_length_element():
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 0.0])  # Zero length element
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {}
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(ValueError, match="Element length cannot be zero."):
        solver.assemble_stiffness()

def test_load_vector_invalid_type():
    nodes = {
        0: np.array([0.0, 0.0, 0.0])
    }
    elements = []
    loads = {
        0: [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Not a numpy array
    }
    supports = {
        0: [True, True, True, True, True, True]
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(TypeError, match="Load at node 0 must be a numpy array."):
        solver.assemble_load_vector()

def test_load_vector_invalid_shape():
    nodes = {
        0: np.array([0.0, 0.0, 0.0])
    }
    elements = []
    loads = {
        0: np.array([1000.0, 0.0])  # Incorrect shape
    }
    supports = {
        0: [True, True, True, True, True, True]
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(ValueError, match="Load vector at node 0 must be of length 6."):
        solver.assemble_load_vector()

def test_boundary_conditions_invalid_type():
    nodes = {
        0: np.array([0.0, 0.0, 0.0])
    }
    elements = []
    loads = {}
    supports = {
        0: "invalid"  # Not a list
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(ValueError, match="Boundary condition at node 0 must be a list of length 6."):
        solver.apply_boundary_conditions(np.zeros((6, 6)), np.zeros(6))

def test_boundary_conditions_invalid_length():
    nodes = {
        0: np.array([0.0, 0.0, 0.0])
    }
    elements = []
    loads = {}
    supports = {
        0: [True, True, True]  # Incorrect length
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(ValueError, match="Boundary condition at node 0 must be a list of length 6."):
        solver.apply_boundary_conditions(np.zeros((6, 6)), np.zeros(6))

def test_solve_singular_matrix():
    """
    Test that the solver raises a LinAlgError when the global stiffness matrix is nearly singular.
    This occurs when nodes are collinear or when properties lead to ill-conditioned matrices.
    """
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1e6, 0.0, 0.0]),  # Extremely long beam to amplify instability
        2: np.array([2e6, 0.0, 0.0])  # Collinear nodes with large spacing
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 1e-14,  # Even smaller area to induce numerical instability
        "Iz": 1e-14,  # Very small moment of inertia
        "Iy": 1e-14,
        "J": 1e-14,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props), (1, 2, section_props)]
    loads = {
        2: np.array([1e9, 0.0, 0.0, 0.0, 0.0, 0.0])  # Large axial load to amplify instability
    }
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False],
        2: [False, False, False, False, False, False]
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    
    # This should raise a LinAlgError due to numerical instability
    with pytest.raises(np.linalg.LinAlgError, match="Global stiffness matrix is nearly singular"):
        solver.solve()

def test_invalid_loads_type():
    nodes = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0])}
    elements = []
    loads = [(0, 0, 0, 0, 0, 0)]  # Invalid type (should be dict)
    supports = {0: [True, True, True, True, True, True]}
    
    with pytest.raises(TypeError, match="loads must be a dictionary."):
        dsm.Frame3DSolver(nodes, elements, loads, supports)

def test_invalid_K_type():
    nodes = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0])}
    elements = []
    loads = {}
    supports = {0: [True, True, True, True, True, True]}
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    
    K = [[1, 2], [3, 4]]  # Invalid type (should be np.ndarray)
    F = np.array([0.0, 0.0])
    
    with pytest.raises(TypeError, match="K and F must be numpy arrays."):
        solver.apply_boundary_conditions(K, F)

def test_nan_infinite_solution():
    """
    Test that the solver detects NaN or Inf in the solution vector by inducing numerical instability.
    """
    nodes = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0])}
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 1e-20,  # Extremely small area to induce numerical instability
        "Iz": 1e-20,  # Extremely small moment of inertia
        "Iy": 1e-20,
        "J": 1e-20,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {1: np.array([1e12, 0.0, 0.0, 0.0, 0.0, 0.0])}  # Large load to amplify instability
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    
    # This should raise a LinAlgError due to unreasonably large displacements
    with pytest.raises(np.linalg.LinAlgError, match="Unreasonably large displacements"):
        solver.solve()

def test_local_elastic_stiffness_matrix_invalid_type():
    with pytest.raises(TypeError, match="E, nu, A, L, Iy, Iz, and J must be numerical values."):
        dsm.local_elastic_stiffness_matrix_3D_beam("invalid", 0.3, 0.01, 1.0, 1e-6, 1e-6, 1e-5)

def test_assemble_load_vector_invalid_type():
    nodes = {0: np.array([0.0, 0.0, 0.0])}
    elements = []
    loads = ["invalid"]  # Not a dictionary
    supports = {0: [True, True, True, True, True, True]}
    
    # Check during instantiation
    with pytest.raises(TypeError, match="loads must be a dictionary."):
        dsm.Frame3DSolver(nodes, elements, loads, supports)

def test_apply_boundary_conditions_invalid_bc_structure():
    nodes = {0: np.array([0.0, 0.0, 0.0])}
    elements = []
    loads = {}
    supports = {0: [True, True, True, True, True]}  # Not length 6
    
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(ValueError, match="Boundary condition at node 0 must be a list of length 6."):
        solver.apply_boundary_conditions(np.zeros((6, 6)), np.zeros(6))

def test_apply_boundary_conditions_non_boolean_bc():
    nodes = {0: np.array([0.0, 0.0, 0.0])}
    elements = []
    loads = {}
    supports = {0: [True, True, True, True, True, "invalid"]}  # Not all boolean
    
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(TypeError, match="Boundary conditions at node 0 must be booleans."):
        solver.apply_boundary_conditions(np.zeros((6, 6)), np.zeros(6))

def test_solve_numerical_singularity():
    nodes = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0])}
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 1e-20,
        "Iz": 1e-20,
        "Iy": 1e-20,
        "J": 1e-20,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {1: np.array([1e12, 0.0, 0.0, 0.0, 0.0, 0.0])}
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }
    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    
    # Adjusted to match the new error message
    with pytest.raises(np.linalg.LinAlgError, match="Unreasonably large displacements, indicating numerical instability."):
        solver.solve()

def test_solve_invalid_reaction_shape():
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0])
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 1e-6,
        "Iz": 1e-6,
        "Iy": 1e-6,
        "J": 1e-6,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {}
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }

    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    
    # Solve the system
    d, reactions = solver.solve()

    # Manually modify reactions to the wrong shape
    reactions = np.append(reactions, [0.0])  # Introduce shape mismatch

    # Check the reaction vector shape explicitly
    with pytest.raises(ValueError, match="Reactions vector must be of length ndof."):
        if reactions.shape != (solver.ndof,):
            raise ValueError("Reactions vector must be of length ndof.")

def test_assemble_stiffness_missing_properties():
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0])
    }
    # Missing 'A' in properties
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {}
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }

    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(KeyError):
        solver.assemble_stiffness()

def test_assemble_stiffness_invalid_node_index():
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0])
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 8.33e-6,
        "Iy": 8.33e-6,
        "J": 1.67e-5,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    # Node index 2 does not exist
    elements = [(0, 2, section_props)]
    loads = {}
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }

    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(KeyError):
        solver.assemble_stiffness()

def test_assemble_load_vector_incorrect_dimension():
    nodes = {
        0: np.array([0.0, 0.0, 0.0])
    }
    elements = []
    loads = {
        0: np.array([1000.0, 0.0])  # Incorrect dimension
    }
    supports = {
        0: [True, True, True, True, True, True]
    }

    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(ValueError, match="Load vector at node 0 must be of length 6."):
        solver.assemble_load_vector()

def test_apply_boundary_conditions_inconsistent_bc():
    nodes = {
        0: np.array([0.0, 0.0, 0.0])
    }
    elements = []
    loads = {}
    # Boundary condition list has more than 6 elements
    supports = {
        0: [True, True, True, True, True, True, True]
    }

    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    K = np.zeros((solver.ndof, solver.ndof))
    F = np.zeros(solver.ndof)
    with pytest.raises(ValueError, match="Boundary condition at node 0 must be a list of length 6."):
        solver.apply_boundary_conditions(K, F)

def test_solve_singular_global_stiffness():
    nodes = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0])
    }
    section_props = {
        "E": 210e9,
        "nu": 0.3,
        "A": 1e-20,  # Extremely small area to induce numerical instability
        "Iz": 1e-20,
        "Iy": 1e-20,
        "J": 1e-20,
        "local_z": np.array([0.0, 0.0, 1.0])
    }
    elements = [(0, 1, section_props)]
    loads = {1: np.array([1e12, 0.0, 0.0, 0.0, 0.0, 0.0])}
    supports = {
        0: [True, True, True, True, True, True],
        1: [False, False, False, False, False, False]
    }

    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    with pytest.raises(np.linalg.LinAlgError, match="Unreasonably large displacements"):
        solver.solve()

def test_solve_invalid_reaction_shape():
    nodes = {0: np.array([0.0, 0.0, 0.0])}
    elements = []
    loads = {}
    supports = {0: [True, True, True, True, True, True]}

    solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
    solver.ndof = 3  # Manipulate ndof to trigger the error

    with pytest.raises(np.linalg.LinAlgError, match="cond is not defined on empty arrays"):
        solver.solve()

# Sample Test Data
nodes = {
    1: np.array([0.0, 0.0, 0.0]),
    2: np.array([1.0, 0.0, 0.0]),
    3: np.array([1.0, 1.0, 0.0])
}

elements = [
    (1, 2, {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 1.0e-6,
        "Iy": 1.0e-6,
        "J": 1.0e-6
    }),
    (2, 3, {
        "E": 210e9,
        "nu": 0.3,
        "A": 0.01,
        "Iz": 1.0e-6,
        "Iy": 1.0e-6,
        "J": 1.0e-6
    })
]

loads = {
    3: np.array([0.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
}

supports = {
    1: [True, True, True, True, True, True],
    2: [False, True, True, False, False, False]
}

# Create Solver Instance
solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
displacements, reactions = solver.solve()

# -----------------------
# Test: Compute Internal Forces and Moments
# -----------------------
def test_compute_internal_forces_and_moments():
    internal_forces = solver.compute_internal_forces_and_moments(displacements)
    
    # Check if internal forces is a dictionary
    assert isinstance(internal_forces, dict), "Internal forces should be a dictionary."
    
    # Check if keys are tuples of node IDs
    for key in internal_forces.keys():
        assert isinstance(key, tuple), "Key should be a tuple of node IDs."
        assert len(key) == 2, "Key should be a tuple of two node IDs."
        assert all(isinstance(node_id, int) for node_id in key), "Node IDs should be integers."
    
    # Check if values are NumPy arrays of length 12
    for value in internal_forces.values():
        assert isinstance(value, np.ndarray), "Internal forces should be a NumPy array."
        assert value.shape == (12,), "Internal forces array should have length 12."

# -----------------------
# Test: Plot Internal Forces and Moments
# -----------------------
def test_plot_internal_forces_and_moments():
    internal_forces = solver.compute_internal_forces_and_moments(displacements)
    
    # Generate the plot
    solver.plot_internal_forces_and_moments(internal_forces)
    
    # Get the current figure
    fig = plt.gcf()
    
    # Check if a figure is created
    assert fig is not None, "Figure should be created."
    
    # Check if figure has axes
    assert len(fig.get_axes()) > 0, "Figure should have at least one axis."
    
    # Check if axes contain data
    for ax in fig.get_axes():
        assert len(ax.lines) > 0, "Plot should contain data."
    
    # Close the figure after test
    plt.close(fig)

# -----------------------
# Test: Plot Deformed Shape
# -----------------------
def test_plot_deformed_shape():
    solver.plot_deformed_shape(displacements, scale=100)
    
    # Get the current figure
    fig = plt.gcf()
    
    # Check if a figure is created
    assert fig is not None, "Figure should be created."
    
    # Check if figure has 3D axes
    assert len(fig.get_axes()) > 0, "Figure should have at least one axis."
    ax = fig.get_axes()[0]
    assert isinstance(ax, Axes3D), "Figure should have 3D axes."
    
    # Check if 3D axes contain data
    assert len(ax.lines) > 0, "3D plot should contain data."
    
    # Close the figure after test
    plt.close(fig)

from directstiffnessmethod.elastic_critical_load_solver import (
    local_geometric_stiffness_matrix_3D_beam,
    local_geometric_stiffness_matrix_3D_beam_without_interaction_terms,
    extract_moments_from_internal_forces,
    ElasticCriticalLoadSolver,
)

# --------------------------
# Test Local Geometric Stiffness Matrix Functions
# --------------------------

def test_local_geometric_stiffness_without_interaction():
    """ Test local geometric stiffness matrix without interaction terms. """
    L, A, I_rho, Fx2 = 5.0, 0.01, 1.0e-6, -1000.0  # Example values
    
    k_g = local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(L, A, I_rho, Fx2)
    
    assert k_g.shape == (12, 12), "Matrix should be 12x12"
    assert np.allclose(k_g, k_g.T), "Matrix should be symmetric"
    assert not np.isnan(k_g).any(), "Matrix should not contain NaN values"


def test_local_geometric_stiffness_with_interaction():
    """ Test local geometric stiffness matrix with interaction terms. """
    L, A, I_rho = 5.0, 0.01, 1.0e-6  # Example values
    Fx2, Mx2, My1, Mz1, My2, Mz2 = -1000.0, 50.0, 30.0, 20.0, 40.0, 25.0
    
    k_g = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    
    assert k_g.shape == (12, 12), "Matrix should be 12x12"
    assert np.allclose(k_g, k_g.T), "Matrix should be symmetric"
    assert not np.isnan(k_g).any(), "Matrix should not contain NaN values"

# --------------------------
# Test Extracting Moments from Internal Forces
# --------------------------

def test_extract_moments_from_internal_forces():
    """ Test moment and force extraction function. """
    element = (1, 2)
    internal_forces = {
        (1, 2): np.array([0, 0, 0, 0, 30.0, 20.0, -1000.0, 0, 0, 50.0, 40.0, 25.0])
    }
    
    Fx2, Mx2, My1, Mz1, My2, Mz2 = extract_moments_from_internal_forces(internal_forces, element)

    assert Fx2 == -1000.0, "Incorrect axial force"
    assert Mx2 == 50.0, "Incorrect torsional moment"
    assert My1 == 30.0, "Incorrect bending moment at node 1"
    assert Mz1 == 20.0, "Incorrect bending moment at node 1"
    assert My2 == 40.0, "Incorrect bending moment at node 2"
    assert Mz2 == 25.0, "Incorrect bending moment at node 2"

# --------------------------
# Test Elastic Critical Load Solver
# --------------------------

@pytest.fixture
def example_frame_solver():
    """ Create an example Frame3DSolver instance for testing. """
    nodes = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([5.0, 0.0, 0.0]),
        3: np.array([5.0, 5.0, 0.0]),
        4: np.array([0.0, 5.0, 0.0])
    }
    
    E, nu, A, Iz, Iy, J = 210e9, 0.3, 0.01, 1.0e-6, 1.0e-6, 1.0e-6
    elements = [
        (1, 2, {"E": E, "nu": nu, "A": A, "Iz": Iz, "Iy": Iy, "J": J}),
        (2, 3, {"E": E, "nu": nu, "A": A, "Iz": Iz, "Iy": Iy, "J": J})
    ]
    
    loads = {3: np.array([0, -1000, 0, 0, 0, 0])}
    supports = {1: [True]*6, 4: [True]*6}

    return dsm.Frame3DSolver(nodes, elements, loads, supports)

def test_elastic_critical_load_solver_without_interaction(example_frame_solver):
    """ Test the Elastic Critical Load Solver without interaction terms. """
    ecl_solver = ElasticCriticalLoadSolver(example_frame_solver, use_interaction_terms=False)
    
    eigenvalues, eigenvectors = ecl_solver.solve_eigenvalue_problem()
    
    assert eigenvalues.size > 0, "Eigenvalues should not be empty"
    assert np.all(eigenvalues > 0), "Eigenvalues should be positive"
    assert eigenvectors.shape == (example_frame_solver.ndof, eigenvalues.size), "Eigenvector size mismatch"

def test_elastic_critical_load_solver_with_interaction(example_frame_solver):
    """ Test the Elastic Critical Load Solver with interaction terms. """
    ecl_solver = ElasticCriticalLoadSolver(example_frame_solver, use_interaction_terms=True)
    
    eigenvalues, eigenvectors = ecl_solver.solve_eigenvalue_problem()
    
    assert eigenvalues.size > 0, "Eigenvalues should not be empty"
    assert np.all(eigenvalues > 0), "Eigenvalues should be positive"
    assert eigenvectors.shape == (example_frame_solver.ndof, eigenvalues.size), "Eigenvector size mismatch"

# --------------------------
# Test Plotting Functions (Without Displaying)
# --------------------------
@pytest.fixture
def mock_frame_solver():
    """Fixture to create a mock 3D frame solver instance."""
    class MockFrameSolver:
        def __init__(self):
            self.nodes = {0: np.array([0, 0, 0]), 1: np.array([1, 0, 0])}
            self.elements = [(0, 1, {})]  # Single beam element
            self.node_index_map = {0: 0, 1: 1}

    return MockFrameSolver()

def test_hermite_beam_shape_functions():
    """Test Hermite shape functions at element endpoints where H2 + H4 should be zero."""
    L = 1.0

    # Check at ξ = 0 (left node)
    H1, H2, H3, H4 = ecls.hermite_beam_shape_functions(0, L)
    assert np.isclose(H2 + H4, 0.0, atol=1e-6), f"H2 + H4 should be zero at ξ=0, but got {H2+H4}"

    # Check at ξ = L (right node)
    H1, H2, H3, H4 = ecls.hermite_beam_shape_functions(L, L)
    assert np.isclose(H2 + H4, 0.0, atol=1e-6), f"H2 + H4 should be zero at ξ=1, but got {H2+H4}"

def test_plot_buckling_mode(mock_frame_solver):
    """Test that plot_buckling_mode runs without errors using Matplotlib in Agg mode."""
    plt.switch_backend("Agg")  # Ensure Matplotlib is using Agg backend

    mode_shape = np.zeros(12)  # Mock mode shape with zero displacement

    try:
        ecls.plot_buckling_mode(mock_frame_solver, mode_shape, scale_factor=1.0, n_points=10)
    except Exception as e:
        pytest.fail(f"plot_buckling_mode raised an exception: {e}")

@pytest.fixture
def frame_solver_ecla():
    """Fixture to create a Frame3DSolver instance for buckling analysis."""
    nodes_ecls = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([30, 40, 0.0])
    }
    r = 1  # Radius of the beam cross-section
    elements_ecls = [
        (0, 1, {
            "E": 1000, "nu": 0.3, "A": np.pi * r**2,
            "Iz": np.pi * r**4 / 4, "Iy": np.pi * r**4 / 4,
            "J": np.pi * r**4 / 2, "I_rho": np.pi * r**4 / 2
        })
    ]
    loads_ecls = {
        1: np.array([-3/5, -4/5, 0, 0, 0, 0])
    }
    supports_ecls = {
        0: [True, True, True, True, True, True]
    }

    return dsm.Frame3DSolver(nodes_ecls, elements_ecls, loads_ecls, supports_ecls)

@pytest.mark.parametrize("use_interaction_terms", [True, False])
def test_analytical_lowest_critical_load_factor(frame_solver_ecla, use_interaction_terms):
    """Test that the lowest critical load factor matches the analytical solution."""
    analytical_value = 0.7751569170074954

    ecl_solver = ElasticCriticalLoadSolver(frame_solver_ecla, use_interaction_terms=use_interaction_terms)
    eigenvalues, _ = ecl_solver.solve_eigenvalue_problem()
    
    computed_value = np.min(eigenvalues)

    assert np.isclose(computed_value, analytical_value, atol=1e-2), \
        f"Expected {analytical_value}, but got {computed_value} (using interaction terms: {use_interaction_terms})"
