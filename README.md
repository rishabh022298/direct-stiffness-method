# 3D Frame Solver using the Direct Stiffness Method

This repository provides a robust and modular implementation of a **3D Frame Solver** using the **Direct Stiffness Method**. It is designed to analyze 3D frames under various loading and boundary conditions using Python and **NumPy**.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Mathematical Formulation](#mathematical-formulation)
- [Installation](#installation)
- [Input Format](#input-format)
- [Output](#output)
- [Error Handling](#error-handling)
- [Acknowledgments](#acknowledgments)

---

## Introduction

The **3D Frame Solver** is designed to:
- Assemble the global stiffness matrix for the structure.
- Apply the boundary conditions.
- Solve the reduced system for nodal displacements and rotations.
- Compute the support reactions.

This solver is particularly useful for engineers, researchers, and students involved in structural analysis of frames in civil, mechanical, and aerospace engineering domains.

---

## Features

- **Local Elastic Stiffness Matrix**: Computes the local stiffness matrix for a 3D beam element considering axial, torsion, and bending effects.
- **Rotation Matrix and Transformation Matrix**: Efficiently calculates the rotation and transformation matrices to handle global and local coordinate transformations.
- **Global Stiffness Assembly**: Assembles the global stiffness matrix using the Direct Stiffness Method.
- **Load Vector Assembly**: Constructs the global load vector for applied forces and moments.
- **Boundary Condition Application**: Flexibly applies various boundary conditions including fixed, pinned, and roller supports.
- **Solution and Post-Processing**:
  - Solves the reduced system of equations for nodal displacements and rotations.
  - Computes support reactions for all constrained degrees of freedom.
- **Error Handling**: Extensive error handling for input validation, numerical stability, and solution accuracy.

---

## Mathematical Formulation

### 1. Local Elastic Stiffness Matrix
The local elastic stiffness matrix for a 3D beam element is computed using:

$$
k_e = 
\begin{bmatrix}
k_{11} & k_{12} \\
k_{21} & k_{22}
\end{bmatrix}
$$

Where:
- $k_{11}, k_{12}, k_{21}, k_{22}$ are sub-matrices representing axial, torsional, and bending stiffness contributions.

### 2. Global Stiffness Matrix
The global stiffness matrix is assembled by transforming the local stiffness matrix into the global coordinate system using:

$$
K_{global} = \Gamma^T k_{local} \Gamma
$$

Where:
- $\Gamma$ is the 12x12 transformation matrix calculated using the rotation matrix for the element.

### 3. Displacement and Reaction Calculation
The system of equations is solved using:

$$
K_{reduced} d_{free} = F_{reduced}
$$

Reactions at the supports are calculated as:

$$
R = K d - F
$$

Where:
- $d$ is the vector of nodal displacements and rotations.
- $F$ is the global load vector.
- $R$ is the reaction vector at constrained degrees of freedom.

---

## Installation

To use the 3D Frame Solver, you need to have **Python** and **NumPy** installed on your system.

### Prerequisites
- Python 3.12
- NumPy

### Installation

**Please ensure that you run the following commands in the terminal after downloading the repository (Please ensure that repository is not in the downloads folder and their relative locations are not changed.)**\
To install the package, first create a virtual environment:
```bash
conda create --name direct-stiffness-env python=3.12
```
Once the environment has been created, activate it:
```bash
conda activate direct-stiffness-env
```
Double check that python is version 3.12 in the environment:
```bash
python --version
```
Please ensure that pip is using the most up to date version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
Create the editable install of the Direct Stiffness Method code (note: you must be in the correct directory, i.e. where all of the files of the repository are.)
```bash
pip install -e .
```
Test the code is working with pytest
```bash
pytest -v --cov=directstiffnessmethod --cov-report term-missing
```
## Input Format
Script can be used for evaluation as follows:

```python
from directstiffnessmethod import direct_stiffness_method as dsm
import numpy as np
```
First user needs to define the nodes and their locations (let's say there are three nodes):
```python
# Define node coordinates (node ID: [x, y, z])
nodes = {
  0: np.array([0.0, 5.0, 0.0]),
  1: np.array([5.0, 0.0, 0.0]),
  2: np.array([5.0, 5.0, 0.0])
}
```
Next, user needs to define material and geometrical properties of various elements. This can be done in the following way (let's say for a 2 element structure):
```python
# Each element is defined as: (node1, node2, section_properties)
section_props_element_1 = {
  "E": 210e9,           # Young's modulus in Pascals
  "nu": 0.3,            # Poisson's ratio
  "A": 0.01,            # Cross-sectional area in m^2
  "Iz": 8.33e-6,        # Moment of inertia about local z axis in m^4
  "Iy": 8.33e-6,        # Moment of inertia about local y axis in m^4
  "J": 1.67e-5,         # Torsional constant in m^4
  "local_z": np.array([0.0, 0.0, 1.0])  # Reference vector for orientation
}

section_props_element_2 = {
  "E": 210e9,           # Young's modulus in Pascals
  "nu": 0.3,            # Poisson's ratio
  "A": 0.01,            # Cross-sectional area in m^2
  "Iz": 8.33e-6,        # Moment of inertia about local z axis in m^4
  "Iy": 8.33e-6,        # Moment of inertia about local y axis in m^4
  "J": 1.67e-5,         # Torsional constant in m^4
  "local_z": np.array([0.0, 0.0, 1.0])  # Reference vector for orientation
}
```
Once the properties haves been defined, then user can move on to finalising the geometry by defining which nodes are connected and what are the properties of those elements.
```python
elements = [
  (0, 2, section_props_element_1),
  (1, 2, section_props_element_2)
]
```
After defining the elements, nodal forces and boundary conditions can be defined in any order.
```python
# For each node, a load vector [Fx, Fy, Fz, Mx, My, Mz] is applied.
loads = {
  2: np.array([0.0, -10000.0, 0.0, 0.0, 0.0, 0.0])  # Applied at node 2 (e.g., vertical load)
}
```
```python
# For each node, provide a list of 6 booleans (True = DOF is fixed).
supports = {
  0: [False, True, True, False, False, True],  # Node 0 is partially fixed
  1: [True, True, True, True, True, True]      # Node 1 is fully fixed
}
```
Once the user is satisfied with the geometry and the applied nodal loads and boundary conditions, then the solver can be initiated:
```python
solver = dsm.Frame3DSolver(nodes, elements, loads, supports)
displacements, reactions = solver.solve()
```
Reshaping and printing the results:
```python
# Create a dictionary for displacements and reactions
disp_dict = {node: disp_matrix[i] for i, node in enumerate(nodes)}
react_dict = {node: reac_matrix[i] for i, node in enumerate(nodes)}
    
# Output the results
print("Nodal Displacements and Rotations:")
for node, disp in disp_dict.items():
  print(f"Node {node}: [u: {disp[0]:.6f}, v: {disp[1]:.6f}, w: {disp[2]:.6f}, "
        f"rot_x: {disp[3]:.6f}, rot_y: {disp[4]:.6f}, rot_z: {disp[5]:.6f}]")
    
print("\nReaction Forces and Moments at Supports:")
for node, react in react_dict.items():
  # Only display reactions for nodes with boundary conditions
  if node in supports:
    print(f"Node {node}: [Fx: {react[0]:.2f}, Fy: {react[1]:.2f}, Fz: {react[2]:.2f}, "
          f"Mx: {react[3]:.2f}, My: {react[4]:.2f}, Mz: {react[5]:.2f}]")
```

## Output
The above script when ran properly will give an output which looks something like this:
```bash
Nodal Displacements and Rotations:
Node 0: [u: 0.000014, v: 0.000000, w: 0.000000, rot_x: 0.000000, rot_y: 0.000000, rot_z: 0.000000]
Node 1: [u: 0.000000, v: 0.000000, w: 0.000000, rot_x: 0.000000, rot_y: 0.000000, rot_z: 0.000000]
Node 2: [u: 0.000014, v: -0.000024, w: -0.000000, rot_x: 0.000000, rot_y: -0.000000, rot_z: -0.000006]

Reaction Forces and Moments at Supports:
Node 0: [Fx: -0.00, Fy: 1.60, Fz: 0.00, Mx: 0.00, My: 0.00, Mz: 6.00]
Node 1: [Fx: 0.00, Fy: 9998.40, Fz: 0.00, Mx: 0.00, My: 0.00, Mz: 2.00]
```
