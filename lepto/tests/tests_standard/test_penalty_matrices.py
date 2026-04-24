import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lepto.standard.model.penalty import create_graph_continuous, create_graph_categorical, categorical_matrix_graph


class TestCreateGraphContinuous:
    @pytest.mark.parametrize("size", [1, 2, 3, 5, 10])
    def test_shape_and_dtype(self, size):
        g = create_graph_continuous(size)
        assert g.shape == (size, size)
        assert g.dtype == float

    @pytest.mark.parametrize("size", [1, 2, 3, 7])
    def test_symmetry(self, size):
        g = create_graph_continuous(size)
        assert_allclose(g, g.T)

    def test_size_1_is_zero(self):
        g = create_graph_continuous(1)
        assert_array_equal(g, np.array([[0.0]]))

    def test_size_2_is_single_edge(self):
        g = create_graph_continuous(2)
        expected = np.array([[0.0, 1.0],
                             [1.0, 0.0]])
        assert_array_equal(g, expected)

    def test_size_4_matches_example(self):
        g = create_graph_continuous(4)
        expected = np.array([[0., 1., 0., 0.],
                             [1., 0., 1., 0.],
                             [0., 1., 0., 1.],
                             [0., 0., 1., 0.]])
        assert_array_equal(g, expected)

    @pytest.mark.parametrize("size", [3, 6])
    def test_only_neighbor_edges_are_one(self, size):
        g = create_graph_continuous(size)
        # Check diagonal zeros
        assert_array_equal(np.diag(g), np.zeros(size))
        # Check that only |i-j| == 1 are ones, others are zero
        for i in range(size):
            for j in range(size):
                if abs(i - j) == 1:
                    assert g[i, j] == 1.0
                else:
                    assert g[i, j] == 0.0


class TestCreateGraphCategorical:
    @pytest.mark.parametrize("size", [1, 2, 4, 10])
    def test_shape_and_dtype(self, size):
        g = create_graph_categorical(size)
        assert g.shape == (size, size)
        assert g.dtype == float

    @pytest.mark.parametrize("size", [1, 2, 5])
    def test_diagonal_zeros(self, size):
        g = create_graph_categorical(size)
        assert_array_equal(np.diag(g), np.zeros(size))

    @pytest.mark.parametrize("size", [2, 4, 7])
    def test_off_diagonal_ones(self, size):
        g = create_graph_categorical(size)
        # all off-diagonal entries should be 1
        mask = ~np.eye(size, dtype=bool)
        assert_array_equal(g[mask], np.ones(mask.sum()))

    def test_size_1(self):
        g = create_graph_categorical(1)
        assert_array_equal(g, np.array([[0.0]]))

    def test_size_4_matches_example(self):
        g = create_graph_categorical(4)
        expected = np.array([[0., 1., 1., 1.],
                             [1., 0., 1., 1.],
                             [1., 1., 0., 1.],
                             [1., 1., 1., 0.]])
        assert_array_equal(g, expected)


class TestCategoricalMatrixGraph:
    def test_size_less_than_2_raises(self):
        with pytest.raises(ValueError, match="size.*at least 2"):
            categorical_matrix_graph(1, None)

    def test_adj_matrix_wrong_shape_raises(self):
        A = np.ones((3, 3))
        with pytest.raises(ValueError, match="adj_matrix.*shape"):
            categorical_matrix_graph(4, A)

    def test_drop_index_out_of_bounds_raises(self):
        A = create_graph_categorical(4)
        with pytest.raises(IndexError):
            categorical_matrix_graph(4, A, drop_index=4)

    def test_adj_with_no_edges_raises(self):
        A = np.zeros((4, 4))
        with pytest.raises(ValueError, match="no non-zero edges"):
            categorical_matrix_graph(4, A)

    def test_none_adj_is_complete_graph_edge_count_and_shape(self):
        size = 5
        D = categorical_matrix_graph(size, None, drop_index=None, normalize=False)
        # complete graph upper triangle edges = size*(size-1)/2
        m = size * (size - 1) // 2
        assert D.shape == (m, size)

    def test_none_adj_complete_graph_rows_are_pairwise_differences(self):
        size = 4
        D = categorical_matrix_graph(size, None, drop_index=None, normalize=False).todense()

        # Expected rows correspond to (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        expected = np.array([
            [ 1, -1,  0,  0],
            [ 1,  0, -1,  0],
            [ 1,  0,  0, -1],
            [ 0,  1, -1,  0],
            [ 0,  1,  0, -1],
            [ 0,  0,  1, -1],
        ], dtype=float)
        assert_array_equal(D, expected)

    def test_drop_index_removes_column(self):
        size = 4
        D = categorical_matrix_graph(size, None, drop_index=0, normalize=False).todense()
        assert D.shape == (size * (size - 1) // 2, size - 1)

        # Compare with undropped then delete col 0
        D_full = categorical_matrix_graph(size, None, drop_index=None, normalize=False).todense()
        expected = np.delete(D_full, 0, axis=1)
        assert_array_equal(D, expected)

    def test_weights_are_applied(self):
        size = 4
        A = np.zeros((size, size))
        # Only edge (0,2) weight 3 and edge (1,3) weight -2
        A[0, 2] = A[2, 0] = 3.0
        A[1, 3] = A[3, 1] = -2.0

        D = categorical_matrix_graph(size, A, drop_index=None, normalize=False).todense()

        # Upper-triangle edges extracted: (0,2) and (1,3) (order per triu_indices mask)
        expected = np.array([
            [ 3,  0, -3,  0],   # 3*(e0 - e2)
            [ 0, -2,  0,  2],   # -2*(e1 - e3) => [-2 at col1, +2 at col3]
        ], dtype=float)
        assert_array_equal(D, expected)

    def test_normalize_scales_before_drop(self):
        size = 5
        # Use complete graph with None adjacency
        D = categorical_matrix_graph(size, None, drop_index=None, normalize=True).todense()
        m = size * (size - 1) // 2
        scale = (size - 1) / m  # as implemented
        D0 = categorical_matrix_graph(size, None, drop_index=None, normalize=False).todense()
        assert_allclose(D, D0 * scale)

    def test_normalize_then_drop_consistent(self):
        size = 5
        D = categorical_matrix_graph(size, None, drop_index=2, normalize=True).todense()

        D0 = categorical_matrix_graph(size, None, drop_index=None, normalize=False).todense()
        m = size * (size - 1) // 2
        scale = (size - 1) / m
        expected = np.delete(D0 * scale, 2, axis=1)
        assert_allclose(D, expected)

    def test_sparse_adj_extracts_only_nonzero_upper_triangle(self):
        size = 5
        A = np.zeros((size, size))
        # upper triangle edges: (0,1),(0,4),(2,3)
        A[0, 1] = A[1, 0] = 1.0
        A[0, 4] = A[4, 0] = 2.0
        A[2, 3] = A[3, 2] = 1.0

        D = categorical_matrix_graph(size, A, drop_index=None, normalize=False).todense()
        assert D.shape == (3, size)

        expected = np.array([
            [ 1, -1,  0,  0,  0],   # (0,1)
            [ 2,  0,  0,  0, -2],   # 2*(0,4)
            [ 0,  0,  1, -1,  0],   # (2,3)
        ], dtype=float)
        assert_array_equal(D, expected)
