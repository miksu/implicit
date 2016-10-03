from __future__ import print_function
import unittest
from scipy.sparse import csr_matrix
import numpy as np
import math

import ctr


class ImplicitALSTest(unittest.TestCase):
    def testImplicit(self):
        regularization = 1e-9
        tolerance = 1e-2

        counts = csr_matrix([[1, 1, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [1, 0, 1, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 0]], dtype=np.float64)

        def check_solution(rows, cols, counts):
            reconstructed = rows.dot(cols.T)
            for i in range(counts.shape[0]):
                for j in range(counts.shape[1]):
                    self.assertTrue(abs(counts[i, j] - reconstructed[i, j]) <
                                    tolerance)

        # check cython version
        rows, cols = ctr.alternating_least_squares(counts * 2, 7,
                                                        regularization,
                                                        use_native=True)
        check_solution(rows, cols, counts.todense())

        # check cython version (using 32 bit factors)
        rows, cols = ctr.alternating_least_squares(counts * 2, 7,
                                                        regularization,
                                                        use_native=True,
                                                        dtype=np.float32)
        check_solution(rows, cols, counts.todense())

        # try out pure python version
        rows, cols = ctr.alternating_least_squares(counts, 7,
                                                        regularization,
                                                        use_native=False)
        check_solution(rows, cols, counts.todense())


    def testCTR(self):
        regularization = 1e-6
        small_topics_regularization = 1e-12
        high_topics_regularization = 1e3
        tolerance = 1e-2

        counts = csr_matrix([[1, 1, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [1, 0, 1, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 0]], dtype=np.float64)

        theta = csr_matrix([[1, 1, 0, 1, 0, 0, 1],
                            [1, 1, 0, 1, 0, 0, 0],
                            [1, 0, 1, 0, 0, 0, 1],
                            [1, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 1, 0, 1, 1],
                            [0, 1, 0, 0, 0, 1, 0]], dtype=np.float64).todense()

        def check_matrix(reconstructed, original):
            assert reconstructed.shape == original.shape
            for i in range(original.shape[0]):
                for j in range(original.shape[1]):
                    expected = original[i, j]
                    got = reconstructed[i, j]
                    self.assertTrue(abs(expected - got) < tolerance,
                        "Expected %s, got %s." % (expected, got))

        def check_solution(rows, cols, counts):
            reconstructed = rows.dot(cols.T)
            check_matrix(reconstructed, counts)

        # check cython version
        rows, cols = ctr.alternating_least_squares(counts * 2, 7,
                                                        regularization,
                                                        topics_regularization=small_topics_regularization,
                                                        iterations=1,
                                                        theta=theta,
                                                        use_native=True)
        check_solution(rows, cols, counts.todense())

        # # check cython version (using 32 bit factors)
        rows, cols = ctr.alternating_least_squares(counts * 2, 7,
                                                    regularization,
                                                    theta=theta,
                                                    topics_regularization=small_topics_regularization,
                                                    use_native=True,
                                                    dtype=np.float32)
        check_solution(rows, cols, counts.todense())

        # try out pure python version
        rows, cols = ctr.alternating_least_squares(counts, 7,
                                                    regularization,
                                                    topics_regularization=small_topics_regularization,
                                                    theta=theta,
                                                    use_native=False)
        check_solution(rows, cols, counts.todense())

        #### topics_regularization high
        rows, cols = ctr.alternating_least_squares(counts, 7,
                                                    regularization,
                                                    topics_regularization=high_topics_regularization,
                                                    theta=theta,
                                                    use_native=False,
                                                    iterations=10)
        check_matrix(cols, theta)


if __name__ == "__main__":
    unittest.main()
