"""
Linear Algebra algorithms and operations.

This module implements fundamental linear algebra operations including
matrix operations, vector operations, and linear system solving.
"""

from typing import List, Tuple, Optional, Union
import math


class Matrix:
    """
    Matrix class with basic linear algebra operations.
    """
    
    def __init__(self, data: List[List[float]]):
        """
        Initialize matrix from 2D list.
        
        Args:
            data: 2D list representing matrix rows
        """
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty")
        
        self.rows = len(data)
        self.cols = len(data[0])
        
        # Ensure all rows have same length
        for row in data:
            if len(row) != self.cols:
                raise ValueError("All rows must have same length")
        
        self.data = [row[:] for row in data]  # Deep copy
    
    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Get element at (row, col)."""
        row, col = key
        return self.data[row][col]
    
    def __setitem__(self, key: Tuple[int, int], value: float):
        """Set element at (row, col)."""
        row, col = key
        self.data[row][col] = value
    
    def __str__(self) -> str:
        """String representation of matrix."""
        return '\n'.join([' '.join(f'{x:8.3f}' for x in row) for row in self.data])
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Matrix addition."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions for addition")
        
        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] 
                 for i in range(self.rows)]
        return Matrix(result)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Matrix subtraction."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have same dimensions for subtraction")
        
        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] 
                 for i in range(self.rows)]
        return Matrix(result)
    
    def __mul__(self, other: Union['Matrix', float]) -> 'Matrix':
        """Matrix multiplication or scalar multiplication."""
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result = [[self.data[i][j] * other for j in range(self.cols)] 
                     for i in range(self.rows)]
            return Matrix(result)
        elif isinstance(other, Matrix):
            # Matrix multiplication
            return matrix_multiply(self, other)
        else:
            raise TypeError("Can only multiply matrix by scalar or another matrix")
    
    def transpose(self) -> 'Matrix':
        """Return transpose of matrix."""
        result = [[self.data[j][i] for j in range(self.rows)] 
                 for i in range(self.cols)]
        return Matrix(result)
    
    def determinant(self) -> float:
        """Calculate determinant of square matrix."""
        return matrix_determinant(self)
    
    def inverse(self) -> Optional['Matrix']:
        """Calculate matrix inverse if it exists."""
        return matrix_inverse(self)
    
    @classmethod
    def identity(cls, n: int) -> 'Matrix':
        """Create n×n identity matrix."""
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return cls(data)
    
    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'Matrix':
        """Create matrix filled with zeros."""
        data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return cls(data)


def vector_dot(v1: List[float], v2: List[float]) -> float:
    """
    Calculate dot product of two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Dot product
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same length")
    
    return sum(a * b for a, b in zip(v1, v2))


def vector_cross(v1: List[float], v2: List[float]) -> List[float]:
    """
    Calculate cross product of two 3D vectors.
    
    Args:
        v1: First 3D vector
        v2: Second 3D vector
        
    Returns:
        Cross product vector
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Cross product requires 3D vectors")
    
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]


def vector_magnitude(v: List[float]) -> float:
    """Calculate magnitude (length) of vector."""
    return math.sqrt(sum(x * x for x in v))


def vector_normalize(v: List[float]) -> List[float]:
    """Normalize vector to unit length."""
    mag = vector_magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize zero vector")
    return [x / mag for x in v]


def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    """
    Multiply two matrices.
    
    Args:
        A: First matrix (m×n)
        B: Second matrix (n×p)
        
    Returns:
        Product matrix (m×p)
        
    Time Complexity: O(mnp)
    Space Complexity: O(mp)
    """
    if A.cols != B.rows:
        raise ValueError("Number of columns in A must equal number of rows in B")
    
    result = Matrix.zeros(A.rows, B.cols)
    
    for i in range(A.rows):
        for j in range(B.cols):
            for k in range(A.cols):
                result[i, j] += A[i, k] * B[k, j]
    
    return result


def matrix_transpose(matrix: Matrix) -> Matrix:
    """
    Transpose a matrix.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Transposed matrix
        
    Time Complexity: O(mn)
    Space Complexity: O(mn)
    """
    return matrix.transpose()


def matrix_determinant(matrix: Matrix) -> float:
    """
    Calculate determinant of square matrix using LU decomposition.
    
    Args:
        matrix: Square matrix
        
    Returns:
        Determinant value
        
    Time Complexity: O(n³)
    Space Complexity: O(n²)
    """
    if matrix.rows != matrix.cols:
        raise ValueError("Determinant requires square matrix")
    
    n = matrix.rows
    
    # Create copy for manipulation
    A = Matrix([row[:] for row in matrix.data])
    det = 1.0
    
    # Gaussian elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k, i]) > abs(A[max_row, i]):
                max_row = k
        
        # Swap rows if needed
        if max_row != i:
            for j in range(n):
                A[i, j], A[max_row, j] = A[max_row, j], A[i, j]
            det *= -1
        
        # Check for zero pivot
        if abs(A[i, i]) < 1e-10:
            return 0.0
        
        det *= A[i, i]
        
        # Eliminate column
        for k in range(i + 1, n):
            factor = A[k, i] / A[i, i]
            for j in range(i, n):
                A[k, j] -= factor * A[i, j]
    
    return det


def matrix_inverse(matrix: Matrix) -> Optional[Matrix]:
    """
    Calculate matrix inverse using Gauss-Jordan elimination.
    
    Args:
        matrix: Square matrix to invert
        
    Returns:
        Inverse matrix if it exists, None otherwise
        
    Time Complexity: O(n³)
    Space Complexity: O(n²)
    """
    if matrix.rows != matrix.cols:
        raise ValueError("Inverse requires square matrix")
    
    n = matrix.rows
    
    # Create augmented matrix [A|I]
    augmented = []
    for i in range(n):
        row = matrix.data[i][:] + [0.0] * n
        row[n + i] = 1.0  # Identity matrix part
        augmented.append(row)
    
    # Gauss-Jordan elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Swap rows if needed
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Check for zero pivot (singular matrix)
        if abs(augmented[i][i]) < 1e-10:
            return None
        
        # Scale pivot row
        pivot = augmented[i][i]
        for j in range(2 * n):
            augmented[i][j] /= pivot
        
        # Eliminate column
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Extract inverse matrix
    inverse_data = []
    for i in range(n):
        inverse_data.append(augmented[i][n:])
    
    return Matrix(inverse_data)


def gaussian_elimination(A: Matrix, b: List[float]) -> Optional[List[float]]:
    """
    Solve linear system Ax = b using Gaussian elimination.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        
    Returns:
        Solution vector x if system has unique solution, None otherwise
        
    Time Complexity: O(n³)
    Space Complexity: O(n²)
    """
    if A.rows != len(b):
        raise ValueError("Matrix and vector dimensions don't match")
    
    n = A.rows
    
    # Create augmented matrix [A|b]
    augmented = []
    for i in range(n):
        augmented.append(A.data[i][:] + [b[i]])
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Swap rows
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Check for zero pivot
        if abs(augmented[i][i]) < 1e-10:
            return None  # No unique solution
        
        # Eliminate column
        for k in range(i + 1, n):
            factor = augmented[k][i] / augmented[i][i]
            for j in range(i, n + 1):
                augmented[k][j] -= factor * augmented[i][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
        x[i] /= augmented[i][i]
    
    return x


def lu_decomposition(matrix: Matrix) -> Tuple[Matrix, Matrix]:
    """
    Perform LU decomposition of a matrix.
    
    Decomposes matrix A into lower triangular L and upper triangular U
    such that A = LU.
    
    Args:
        matrix: Square matrix to decompose
        
    Returns:
        Tuple (L, U) of lower and upper triangular matrices
        
    Time Complexity: O(n³)
    Space Complexity: O(n²)
    """
    if matrix.rows != matrix.cols:
        raise ValueError("LU decomposition requires square matrix")
    
    n = matrix.rows
    L = Matrix.zeros(n, n)
    U = Matrix.zeros(n, n)
    
    for i in range(n):
        # Upper triangular matrix U
        for k in range(i, n):
            sum_val = sum(L[i, j] * U[j, k] for j in range(i))
            U[i, k] = matrix[i, k] - sum_val
        
        # Lower triangular matrix L
        for k in range(i, n):
            if i == k:
                L[i, i] = 1.0  # Diagonal elements of L are 1
            else:
                sum_val = sum(L[k, j] * U[j, i] for j in range(i))
                L[k, i] = (matrix[k, i] - sum_val) / U[i, i]
    
    return L, U


def qr_decomposition(matrix: Matrix) -> Tuple[Matrix, Matrix]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        matrix: Matrix to decompose
        
    Returns:
        Tuple (Q, R) where Q is orthogonal and R is upper triangular
        
    Time Complexity: O(mn²)
    Space Complexity: O(mn)
    """
    m, n = matrix.rows, matrix.cols
    Q = Matrix.zeros(m, n)
    R = Matrix.zeros(n, n)
    
    for j in range(n):
        # Get j-th column of A
        v = [matrix[i, j] for i in range(m)]
        
        # Subtract projections onto previous columns
        for i in range(j):
            q_i = [Q[k, i] for k in range(m)]
            proj = vector_dot(v, q_i)
            R[i, j] = proj
            
            for k in range(m):
                v[k] -= proj * q_i[k]
        
        # Normalize
        norm = vector_magnitude(v)
        R[j, j] = norm
        
        if norm > 1e-10:
            for i in range(m):
                Q[i, j] = v[i] / norm
    
    return Q, R


def eigenvalues_2x2(matrix: Matrix) -> List[complex]:
    """
    Calculate eigenvalues of 2×2 matrix analytically.
    
    Args:
        matrix: 2×2 matrix
        
    Returns:
        List of eigenvalues (may be complex)
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if matrix.rows != 2 or matrix.cols != 2:
        raise ValueError("This function only works for 2×2 matrices")
    
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    
    # Characteristic polynomial: λ² - trace*λ + det = 0
    trace = a + d
    det = a * d - b * c
    
    # Discriminant
    discriminant = trace * trace - 4 * det
    
    if discriminant >= 0:
        sqrt_disc = math.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2
        return [complex(lambda1, 0), complex(lambda2, 0)]
    else:
        sqrt_disc = math.sqrt(-discriminant)
        real_part = trace / 2
        imag_part = sqrt_disc / 2
        return [complex(real_part, imag_part), complex(real_part, -imag_part)]
