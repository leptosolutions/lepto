
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sp
from typing import Optional

def create_graph_continuous(size):
    
    """
    Create a simple first‑order continuous adjacency matrix.

    The returned matrix is a square matrix of shape (size, size) where each
    element indicates adjacency between consecutive indices. Concretely:
    • graph[i, i+1] = 1
    • graph[i+1, i] = 1
    • all other entries are 0

    This structure corresponds to a linear chain graph where each node is
    connected to its immediate neighbor.

    Parameters
    ----------
    size : int
        Number of nodes in the graph. Must be >= 1.

    Returns
    -------
    graph : ndarray of shape (size, size)
        Symmetric adjacency matrix representing a continuous (chain) graph.

    Examples
    --------
    >>> create_graph_continuous(4)
    array([[0., 1., 0., 0.],
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [0., 0., 1., 0.]])
    """
    graph = np.zeros((size, size), dtype=float)
    np.fill_diagonal(graph[:, 1:], 1)
    np.fill_diagonal(graph[1:, :], 1)
    return graph

def create_graph_categorical(size): 
    """
    Create a fully connected categorical adjacency matrix.

    The returned matrix is a square array where all categories are considered
    mutually connected except with themselves. Concretely:
      • graph[i, j] = 1 for all i ≠ j
      • graph[i, i] = 0

    This structure represents a categorical penalty graph where each category
    differs independently from every other category.

    Parameters
    ----------
    size : int
        Number of categories. Must be >= 1.

    Returns
    -------
    graph : ndarray of shape (size, size)
        Symmetric adjacency matrix with zeros on the diagonal and ones elsewhere.

    Examples
    --------
    >>> create_graph_categorical(4)
    array([[0., 1., 1., 1.],
           [1., 0., 1., 1.],
           [1., 1., 0., 1.],
           [1., 1., 1., 0.]])
    """

    graph = np.ones((size, size), dtype=float)
    np.fill_diagonal(graph, 0)
    return graph

def categorical_matrix_graph(
    size: int,
    adj_matrix: Optional[np.ndarray],
    drop_index: Optional[int] = 0,
    *,
    normalize: bool = False
) -> np.ndarray:
    """
    Generate a graph-guided penalty matrix (dense) for a categorical variable.

    Each row encodes a pairwise difference between levels connected in the graph:
        row for edge (i, j), i < j:  e[i] - e[j]
    Optionally weighted by the adjacency matrix entry A[i, j].

    Construction (vectorized):
      - Extract upper-triangular (i, j) indices of adjacency (i < j, off-diagonal)
      - Build a dense matrix with +1 at column i and -1 at column j for each edge-row
      - Multiply rows by edge weights (if provided)
      - Optionally drop one column (typical one-hot constraint)

    Shapes
    ------
    - Without column drops: rows = number of edges in upper triangle, cols = size
    - With a dropped column: rows = same, cols = size - 1

    Parameters
    ----------
    size : int
        Number of levels/bins (columns). Must be >= 2.
    adj_matrix : ndarray or None
        Symmetric adjacency/weight matrix of shape (size, size).
        Only the upper-triangular (k=1) entries are used as edges (i<j).
        If None, uses the complete graph (all pairs i<j with weight 1).
    drop_index : int or None, optional (default=0)
        Column index to drop from the final matrix. If None, no column is dropped.
        Must satisfy 0 <= drop_index < size when provided.
    normalize : bool, optional (default=True)
        If True, scales the matrix by `(num_cols - 1) / num_rows` before any column drop.
        For a complete graph this equals `2 / size`. With arbitrary adjacency, we use
        `(C - 1)/R` where `C`=num_cols and `R`=num_rows.

    Returns
    -------
    D : ndarray
        Dense graph-guided penalty matrix:
          - shape is (R, size) if `drop_index is None`
          - shape is (R, size-1) if `drop_index` is an integer
        where R is the number of edges extracted from `adj_matrix` (upper triangle).

    Raises
    ------
    ValueError
        If `size < 2`, or `adj_matrix` shape is not (size, size), or `drop_index`
        is out of bounds, or adjacency has no edges.
    """
    if size < 2:
        raise ValueError("`size` must be at least 2.")

    iu, ju = np.triu_indices(size, k=1)
    if adj_matrix is None:
        i, j = iu, ju
        weights = np.ones(i.shape, dtype=np.float32)
    else:
        if sp.issparse(adj_matrix):
            A = adj_matrix.toarray().astype(np.float32)
        else:
            A = np.asarray(adj_matrix, dtype=np.float32)
        if A.shape != (size, size):
            raise ValueError("`adj_matrix` must have shape (size, size).")
        w = A[iu, ju]
        mask = w != 0
        i, j = iu[mask], ju[mask]
        weights = w[mask].astype(np.float32)
        if i.size == 0:
            raise ValueError("Adjacency has no non-zero edges in upper triangle (k=1).")

    m = i.size
    n_cols = size - 1 if drop_index is not None else size
    D = lil_matrix((m, n_cols), dtype=np.float32)

    for row, (ii, jj, weight) in enumerate(zip(i, j, weights)):
        col_i = ii if drop_index is None or ii < drop_index else ii - 1
        col_j = jj if drop_index is None or jj < drop_index else jj - 1
        if drop_index is None or ii != drop_index:
            D[row, col_i] = weight
        if drop_index is None or jj != drop_index:
            D[row, col_j] = -weight

    if normalize:
        D *= (size - 1) / m

    return D.tocsr()

