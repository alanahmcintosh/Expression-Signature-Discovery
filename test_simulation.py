import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd
from simulation_functions import  *

class TestSimulations(unittest.TestCase):

    def test_sim_mut_shape_and_type(self):
        """
        Check if output shape is correct, all values are 0 or 1.
        """
        probs = [0.1, 0.5, 0.9]
        n = 1000
        mut = sim_mut(probs, n)

        self.assertEqual(mut.shape, (n, len(probs)))
        self.assertTrue(np.all((mut == 0) | (mut == 1)))

    def test_sim_mut_approx_means(self):
        """
        Checks mean mutation rates are close to the input probabilities.
        """
        probs = [0.1, 0.5, 0.9]
        n = 10000
        mut = sim_mut(probs, n)
        means = mut.mean(axis=0)

        # Within 5% absolute error
        for i, p in enumerate(probs):
            self.assertAlmostEqual(means[i], p, delta=0.05)

    def test_sim_cna_shape_and_nonnegativity(self):
        '''
        Checks if output shape is correct and
        all values are integers.
        '''
        lambdas = [1, 5, 10]
        n = 1000
        cna = sim_cna(lambdas, n)

        self.assertEqual(cna.shape, (n, len(lambdas)))
        self.assertTrue(np.issubdtype(cna.dtype, np.integer))

    def test_sim_cna_poisson_means(self):
        """
        Checks if output means are close to the input lambdas.
        """
        lambdas = [1, 5, 10]
        n = 10000
        cna = sim_cna(lambdas, n)
        means = cna.mean(axis=0)

        for i, lam in enumerate(lambdas):
            self.assertAlmostEqual(means[i], lam, delta=0.2)

    def test_estimate_deseq2_output_shapes(self):
        """
        Checks output shapes are correct and all values are non-negative.
        """
        df = pd.DataFrame(np.random.poisson(10, size=(50, 20)))
        gene_means, gene_vars, D, s, mu_matrix = estimate_deseq2_parameters(df, seed=42)

        self.assertEqual(gene_means.shape[0], df.shape[1])
        self.assertEqual(gene_vars.shape[0], df.shape[1])
        self.assertEqual(D.shape[0], df.shape[1])
        self.assertEqual(s.shape[0], df.shape[0])
        self.assertEqual(mu_matrix.shape, df.shape)
        self.assertTrue((mu_matrix > 0).all().all())

    def test_sim_mod_exp_shapes(self):
        """
        Checsk output shapes are correct and all values are non-negative.
        """
        n_samples, n_alts, n_genes = 100, 5, 50
        x = np.random.binomial(1, 0.2, size=(n_samples, n_alts))
        B = np.random.normal(0, 1, size=(n_alts, n_genes))
        C = np.random.poisson(2, size=(n_samples, n_genes))
        D = np.random.gamma(1, 1, size=n_genes)
        s = np.random.lognormal(0, 0.2, size=n_samples)
        a = np.random.normal(5, 1, size=n_genes)

        mu = sim_mod_exp(x, B, C, D, s, a)

        self.assertEqual(mu.shape, (n_samples, n_genes))
        self.assertTrue(np.all(mu >= 0))

    def test_sample_nb_shapes_and_nonnegativity(self):
        """
        Checks output shapes and non-negativity of output.
        """
        mu = np.abs(np.random.normal(10, 3, size=(100, 50)))
        disp = np.abs(np.random.normal(1, 0.5, size=50))
        counts = sample_nb(mu, disp)

        self.assertEqual(counts.shape, mu.shape)
        self.assertTrue(np.all(counts >= 0))
        self.assertTrue(np.issubdtype(counts.dtype, np.integer))

    def test_sim_mod_exp_mut_and_cna_effects(self):
        """
        Tests that mutations and CNA values alter expression appropriately.
        """
        # Simple test case with 3 samples and 2 genes
        x_mut = np.array([[0, 0],
                        [1, 0],
                        [0, 1]])  # shape (3, 2)

        B = np.array([[2, 0],  # alteration 0 increases gene 0 by 2
                    [0, 3]])  # alteration 1 increases gene 1 by 3

        C = np.array([[1, 1],   # no CNA effect
                    [1, 1],   # no CNA effect
                    [1, 2]])  # sample 2 has double copy number for gene 1

        D = np.array([1, 1])  # not used in mu, placeholder

        s = np.array([1, 1, 1])  # no library size scaling

        a = np.array([10, 10])  # baseline expression for both genes

        mu = sim_mod_exp(x_mut, B, C, D, s, a)

        expected_mu = np.array([
            [10, 10],       # No alteration, baseline only
            [12, 10],       # Alteration 0 adds 2 to gene 0
            [10, 26]        # Alteration 1 adds 3 to gene 1 → (10+3)*2 = 26
        ])

        assert_array_equal(mu, expected_mu)
