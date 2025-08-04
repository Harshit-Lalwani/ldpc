import numpy as np
import scipy.sparse as sp
from typing import Union


def quasi_cyclic_ldpc_matrix(base_matrix: Union[np.ndarray, list], block_size: int) -> sp.csr_matrix:
    """
    Create a quasi-cyclic LDPC matrix from a base matrix and block size.
    
    The quasi-cyclic LDPC matrix is constructed by replacing each entry in the base matrix
    with a circulant matrix of the specified block size. If an entry is -1, it becomes
    a zero block. Otherwise, the entry specifies the circular shift for the identity matrix.
    
    Parameters
    ----------
    base_matrix : Union[np.ndarray, list]
        The base matrix where each entry represents either a circulant shift (non-negative integer)
        or a zero block (-1). Can be a numpy array or nested list.
    block_size : int
        The size of each circulant block (both dimensions).
        
    Returns
    -------
    sp.csr_matrix
        The quasi-cyclic LDPC matrix in sparse CSR format.
        
    Raises
    ------
    TypeError
        If the input variables are not of the correct types.
    ValueError
        If block_size is not positive or if base_matrix contains invalid values.
        
    Examples
    --------
    >>> # Simple 2x2 base matrix with block size 3
    >>> base = [[0, 1], [2, -1]]
    >>> qc_matrix = quasi_cyclic_ldpc_matrix(base, 3)
    >>> print(qc_matrix.shape)
    (6, 6)
    
    >>> # Your specific base matrix P
    >>> P = [[16, 17, 22, 24, 9, 3, 14, -1, 4, 2, 7, -1, 26, -1, 2, -1, 21, -1, 1, 0, -1, -1, -1, -1],
    ...      [25, 12, 12, 3, 3, 26, 6, 21, -1, 15, 22, -1, 15, -1, 4, -1, -1, 16, -1, 0, 0, -1, -1, -1],
    ...      [25, 18, 26, 16, 22, 23, 9, -1, 0, -1, 4, -1, 4, -1, 8, 23, 11, -1, -1, -1, 0, 0, -1, -1],
    ...      [9, 7, 0, 1, 17, -1, -1, 7, 3, -1, 3, 23, -1, 16, -1, -1, 21, -1, 0, -1, -1, 0, 0, -1],
    ...      [24, 5, 26, 7, 1, -1, -1, 15, 24, 15, -1, 8, -1, 13, -1, 13, -1, 11, -1, -1, -1, -1, 0, 0],
    ...      [2, 2, 19, 14, 24, 1, 15, 19, -1, 21, -1, 2, -1, 24, -1, 3, -1, 2, 1, -1, -1, -1, -1, 0]]
    >>> pcm = quasi_cyclic_ldpc_matrix(P, 27)
    >>> print(f"Matrix shape: {pcm.shape}")
    Matrix shape: (162, 648)
    """
    
    # Type checking
    if not isinstance(block_size, int):
        raise TypeError("The input variable 'block_size' must be of type 'int'.")
    
    if block_size <= 0:
        raise ValueError("Block size must be positive.")
    
    # Convert base_matrix to numpy array if it's a list
    if isinstance(base_matrix, list):
        base_matrix = np.array(base_matrix)
    elif not isinstance(base_matrix, np.ndarray):
        raise TypeError("The input variable 'base_matrix' must be a numpy array or list.")
    
    # Validate base matrix entries
    if not np.all((base_matrix >= 0) | (base_matrix == -1)):
        raise ValueError("Base matrix entries must be non-negative integers or -1.")
    
    base_rows, base_cols = base_matrix.shape
    total_rows = base_rows * block_size
    total_cols = base_cols * block_size
    
    # Lists to store the sparse matrix data
    row_indices = []
    col_indices = []
    data = []
    
    # Process each block in the base matrix
    for i in range(base_rows):
        for j in range(base_cols):
            shift = base_matrix[i, j]
            
            if shift != -1:  # Non-zero block
                # Normalize shift to be within [0, block_size)
                shift = shift % block_size
                
                # Create circulant matrix entries
                for k in range(block_size):
                    row_pos = i * block_size + k
                    col_pos = j * block_size + ((k + shift) % block_size)
                    
                    row_indices.append(row_pos)
                    col_indices.append(col_pos)
                    data.append(np.uint8(1))
    
    # Create the sparse matrix
    return sp.csr_matrix(
        (data, (row_indices, col_indices)), 
        shape=(total_rows, total_cols), 
        dtype=np.uint8
    )


def ldpc_quasi_cyclic_matrix(block_size: int, base_matrix: Union[np.ndarray, list]) -> sp.csr_matrix:
    """
    Alternative function name matching MATLAB convention for compatibility.
    
    This is an alias for quasi_cyclic_ldpc_matrix with parameter order matching
    the MATLAB function ldpcQuasiCyclicMatrix(blockSize, P).
    
    Parameters
    ----------
    block_size : int
        The size of each circulant block.
    base_matrix : Union[np.ndarray, list]
        The base matrix.
        
    Returns
    -------
    sp.csr_matrix
        The quasi-cyclic LDPC matrix in sparse CSR format.
        
    Examples
    --------
    >>> P = [[0, 1], [2, -1]]
    >>> pcm = ldpc_quasi_cyclic_matrix(3, P)
    >>> print(pcm.shape)
    (6, 6)
    """
    return quasi_cyclic_ldpc_matrix(base_matrix, block_size)
