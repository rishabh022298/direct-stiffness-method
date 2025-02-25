from directstiffnessmethod import direct_stiffness_method as dsm
import numpy as np
from pathlib import Path

# -----------------------
# Defining Nodes
# -----------------------
"""
syntax: 
nodes = {
        node ID: np.array([x, y, z])
    }
Even if the problem is set in 2D, still all 3 coordinates are required.
Few nodes with coordinates [0, 0, 0] are provided. Define all of the nodes using the similar syntax.
"""
nodes = {
        0: np.array([0, 5, 0]),
        1: np.array([5, 0, 0]),
        2: np.array([5, 5, 0])
        # Add more nodes here by inserting , at the end of the previous line
    }

# -----------------------
# Defining Element Properties
# -----------------------
"""
Sample code for defining element properties is provided. Edit the properties if user wants to use something different.
User can also copy and paste the following code and rename the variable to define properties for more elements.
Note for the user: Please calculate each quantity like area and moment of inertia before entering the values.
This code cannot handle deriving the quantities itself.
"""
section_props_element_1 = {
  "E": 210e9,           # Young's modulus
  "nu": 0.3,            # Poisson's ratio
  "A": 0.01,            # Cross-sectional area
  "Iz": 8.33e-6,        # Moment of inertia about local z axis
  "Iy": 8.33e-6,        # Moment of inertia about local y axis
  "J": 1.67e-5,         # Torsional constant
  "local_z": np.array([0.0, 0.0, 1.0])  # Reference vector for orientation
}

section_props_element_2 = {
  "E": 210e9,           # Young's modulus
  "nu": 0.3,            # Poisson's ratio
  "A": 0.01,            # Cross-sectional area
  "Iz": 8.33e-6,        # Moment of inertia about local z axis
  "Iy": 8.33e-6,        # Moment of inertia about local y axis
  "J": 1.67e-5,         # Torsional constant in m^4
  "local_z": np.array([0.0, 0.0, 1.0])  # Reference vector for orientation
}

# Add more element properties in the space provided below





# -----------------------
# Defining Elements
# -----------------------
"""
Elements can be initialized as:
elements = [
  (nodeID1, nodeID2, section_props_element_1),  # first element
  (nodeID1, nodeID2, section_props_element_2)   # second element 
]
Sample code has been provided for elements with node 0 to node 2 and node 1 to node 2. User can edit the following code and add more elements.
"""
elements = [
  (0, 2, section_props_element_1),
  (1, 2, section_props_element_2)
  # Add more elements here using the proper nodeID and the element properties. Please don't forget to add a , at the end of the previous line if new elements are being added
]

# -----------------------
# Applying Nodal Loads
# -----------------------
"""
For each node where user wants to apply a load, a load vector with 6 inputs of the following form [Fx, Fy, Fz, Mx, My, Mz] can be applied.
A sample is provided. User can add more.
Syntax:
loads = {
  nodeID: np.array([Fx, Fy, Fz, Mx, My, Mz])
}
"""
loads = {
  2: np.array([0.0, -10000.0, 0.0, 0.0, 0.0, 0.0])  # Applied at node 2 (e.g., vertical load)
  # User can add more nodal loads in the similar fashion. Please don't forget to add , at the end of the previous line if you are adding more nodes with loads
}


# -----------------------
# Applying Boundary Conditions
# -----------------------
"""
Boundary conditions at any node are provided in the form of 6 booleans (True = Fixed DOF).
First boolean is for displacement in x.
Second boolean is for displacement in y.
Third boolean is for displacement in z.
Fourth boolean is for rotation along x.
Fifth boolean is for rotation along y.
Sixth boolean is for rotation along z.
Syntax:
supports = {
  nodeID: [True, True, True, True, True, True], # Completely fixed
  nodeID: [False, True, True, False, False, True] # Motion allowed in x direction and rotation allowed along the x and y axis
}
The 6 components are [displacement in x direction, displacement in y direction, displacement in z direction, rotation about x, rotation about y, rotation about z]
Sample conditions are provided. User can edit them and add more as per the requirements.
"""
supports = {
  0: [False, True, True, False, False, True],  # Node 0 is partially fixed
  1: [True, True, True, True, True, True]      # Node 1 is fully fixed
  # USer can add more boundary conditions here. Please remember to insert , at the end of the previous line while adding a new node with a particular boundary condition.
}


# -----------------------
# Initiating the Solver and Printing the Results
# -----------------------
"""
If user hasn't changed the name of the variables then code does not require any further changes or additions from this point.
User can run the script after saving from the terminal using the following command:

python example.py
"""
solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
displacements, reactions = solver.solve()
disp_matrix = displacements.reshape((-1, 6))
reac_matrix = reactions.reshape((-1, 6))
# Create a dictionary for displacements and reactions
disp_dict = {node: disp_matrix[i] for i, node in enumerate(nodes)}
react_dict = {node: reac_matrix[i] for i, node in enumerate(nodes)}
    
# Output the results
print("Nodal Displacements and Rotations:")
for node, disp in disp_dict.items():
  print(f"Node {node}: [u: {disp[0]:.10f}, v: {disp[1]:.10f}, w: {disp[2]:.10f}, "
        f"rot_x: {disp[3]:.10f}, rot_y: {disp[4]:.10f}, rot_z: {disp[5]:.10f}]")
    
print("\nReaction Forces and Moments at Supports:")
for node, react in react_dict.items():
  # Only display reactions for nodes with boundary conditions
  if node in supports:
    print(f"Node {node}: [Fx: {react[0]:.10f}, Fy: {react[1]:.10f}, Fz: {react[2]:.10f}, "
          f"Mx: {react[3]:.10f}, My: {react[4]:.10f}, Mz: {react[5]:.10f}]")

