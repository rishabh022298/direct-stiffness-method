# 3D Frame Solver using the Direct Stiffness Method

This repository provides a robust and modular implementation of a **3D Frame Solver** using the **Direct Stiffness Method**. It is designed to analyze 3D frames under various loading and boundary conditions using Python and **NumPy**.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Mathematical Formulation](#mathematical-formulation)
- [Installation](#installation)
- [Usage](#usage)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Examples](#examples)
- [Error Handling](#error-handling)
- [Contributing](#contributing)
- [License](#license)
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
- $$k_{11}, k_{12}, k_{21}, k_{22}$$ are sub-matrices representing axial, torsional, and bending stiffness contributions.

### 2. Global Stiffness Matrix
The global stiffness matrix is assembled by transforming the local stiffness matrix into the global coordinate system using:

$$
K_{global} = \Gamma^T k_{local} \Gamma
$$

Where:
- $ \Gamma $ is the 12x12 transformation matrix calculated using the rotation matrix for the element.

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
- $ d $ is the vector of nodal displacements and rotations.
- $ F $ is the global load vector.
- $ R $ is the reaction vector at constrained degrees of freedom.

---

## Installation

To use the 3D Frame Solver, you need to have **Python** and **NumPy** installed on your system.

### Prerequisites
- Python 3.x
- NumPy

### Installation via pip

```bash
pip install numpy
