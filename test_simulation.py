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

    def test_sim_cna_output_shape_and_type(self):
        means = np.array([1.0, 2.0, 3.0])
        vars_ = np.array([2.0, 4.0, 10.0])
        n_samples = 100

        sim = sim_cna(means, vars_, n_samples)

        self.assertEqual(sim.shape, (n_samples, len(means)))
        self.assertTrue(np.all(sim >= 0))
        self.assertTrue(np.issubdtype(sim.dtype, np.floating))  # can be float since Poisson-like NB

    def test_sim_cna_mean_and_variance_approximation(self):
        means = np.array([2.0, 4.0])
        vars_ = np.array([5.0, 10.0])
        n_samples = 10000

        sim = sim_cna(means, vars_, n_samples)
        sim_means = sim.mean(axis=0)
        sim_vars = sim.var(axis=0)

        for i in range(len(means)):
            self.assertAlmostEqual(sim_means[i], means[i], delta=0.2 * means[i])
            self.assertAlmostEqual(sim_vars[i], vars_[i], delta=0.2 * vars_[i])


    def test_sim_by_cluster_balanced_output(self):
        import warnings

        # Create mock data
        samples = [f"S{i}" for i in range(10)]
        genes = [f"G{i}" for i in range(5)]
        mut = pd.DataFrame(np.random.binomial(1, 0.3, size=(10, 5)), index=samples, columns=genes)
        cna = pd.DataFrame(np.random.normal(1.0, 0.5, size=(10, 5)), index=samples, columns=genes)
        subtype = pd.DataFrame({'Subtype': ['A'] * 5 + ['B'] * 5}, index=samples)

        n_samples = 100

        # Validate each cluster's NB parameter calculations
        for subtype_label in subtype['Subtype'].unique():
            C_s = cna[subtype['Subtype'] == subtype_label]
            means = C_s.mean()
            variances = C_s.var()

            for i, gene in enumerate(C_s.columns):
                mu = means[gene]
                var = variances[gene]

                if var <= mu or not np.isfinite(var):
                    var = mu + max(1e-3, mu * 0.05)

                r = mu**2 / (var - mu) if (var - mu) > 0 else np.nan
                p = r / (r + mu) if r + mu > 0 else np.nan

                # Assert r and p are valid
                self.assertTrue(np.isfinite(r) and r > 0, msg=f"Invalid r for gene {gene}: r={r}, mu={mu}, var={var}")
                self.assertTrue(0 < p < 1, msg=f"Invalid p for gene {gene}: p={p}, r={r}, mu={mu}")

        # Run the actual simulation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress scipy/pydeseq2 warnings for clean test output
            sim_mut, sim_cna_out = sim_by_cluster(mut, cna, subtype, n_samples)

        self.assertEqual(sim_mut.shape, (n_samples, mut.shape[1]))
        self.assertEqual(sim_cna_out.shape, (n_samples, cna.shape[1]))
        self.assertTrue(((sim_mut.values == 0) | (sim_mut.values == 1)).all())
        self.assertTrue(np.all(np.isfinite(sim_cna_out.values)))


    def test_sim_by_cluster_proportionality(self):
        # Uneven subtypes: 3 in A, 7 in B
        samples = [f"S{i}" for i in range(10)]
        subtype_labels = ['A'] * 3 + ['B'] * 7
        subtype = pd.DataFrame({'Subtype': subtype_labels}, index=samples)

        mut = pd.DataFrame(np.random.binomial(1, 0.5, size=(10, 4)), index=samples, columns=[f"G{i}" for i in range(4)])
        cna = pd.DataFrame(np.random.normal(1.0, 0.2, size=(10, 4)), index=samples, columns=[f"G{i}" for i in range(4)])

        n_samples = 100
        _, _ = sim_by_cluster(mut, cna, subtype, n_samples)  # checks that rounding adjustment works without error



    def test_estimate_deseq2_parameters_output(self):
        df = pd.DataFrame(np.random.poisson(10, size=(50, 20)), columns=[f"G{i}" for i in range(20)])
        gene_means, gene_vars, dispersions, size_factors = estimate_deseq2_parameters(df, seed=42)

        self.assertEqual(gene_means.shape[0], df.shape[1])
        self.assertEqual(gene_vars.shape[0], df.shape[1])
        self.assertEqual(dispersions.shape[0], df.shape[1])
        self.assertEqual(size_factors.shape[0], df.shape[0])
        self.assertTrue(np.all(dispersions > 0))

    def test_simulate_rna_background_shape(self):
        gene_means = pd.Series([10, 20, 30], index=["A", "B", "C"])
        size_factors = np.array([1.0, 0.5])
        bg = simulate_rna_background(gene_means, dispersions=None, size_factors=size_factors, n_samples=2)

        self.assertEqual(bg.shape, (2, 3))
        self.assertTrue((bg.values >= 0).all())

    def test_generate_signatures_structure(self):
        genes = [f"G{i}" for i in range(50)]
        alts = [f"G{i}_mut" for i in range(10)]

        sigs = generate_signatures(genes, alts, min_size=5, max_size=5)
        
        for alt, sig in sigs.items():
            self.assertIn(alt.split('_')[0], sig['targets'])
            self.assertEqual(len(sig['targets']), 5)
            self.assertEqual(set(sig['effects'].keys()), set(sig['targets']))
            self.assertTrue(all(isinstance(v, float) for v in sig['effects'].values()))

    
    def test_inject_expression_effects_additive_and_cna(self):
        expr_df = pd.DataFrame([[10, 10], [10, 10]], columns=["G1", "G2"], index=["S1", "S2"])
        alts = pd.DataFrame([[1], [0]], columns=["G1_mut"], index=["S1", "S2"])
        sigs = {
            "G1_mut": {
                "targets": ["G1"],
                "effects": {"G1": 2.0}
            }
        }
        cna = pd.DataFrame([[2, 1], [1, 1]], columns=["G1", "G2"], index=["S1", "S2"])

        mod = inject_expression_effects(expr_df, alts, sigs, cna_df=cna)
        
        expected = pd.DataFrame([[24.0, 10], [10, 10]], columns=["G1", "G2"], index=["S1", "S2"])
        assert_array_almost_equal(mod.values, expected.values)

    def test_sample_nb_properties(self):
        mu = np.full((100, 10), 20.0)
        dispersions = pd.Series(np.full(10, 0.1))
        counts = sample_nb(mu, dispersions)

        self.assertEqual(counts.shape, mu.shape)
        self.assertTrue((counts >= 0).all())
        self.assertTrue(np.issubdtype(counts.dtype, np.integer))

    def test_simulate_rna_with_signatures_end_to_end(self):
        np.random.seed(0)
        genes = [f"G{i}" for i in range(50)]
        samples = [f"S{i}" for i in range(10)]
        
        rna_df = pd.DataFrame(np.random.poisson(10, size=(10, 50)), index=samples, columns=genes)
        alteration_df = pd.DataFrame(np.random.binomial(1, 0.2, size=(10, 5)), index=samples, columns=[f"G{i}_mut" for i in range(5)])
        cna_df = pd.DataFrame(np.random.normal(1.0, 0.2, size=(10, 50)), index=samples, columns=genes)

        sim_expr, sigs = simulate_rna_with_signatures(rna_df, alteration_df, cna_df, n_samples=10, seed=123)

        self.assertEqual(sim_expr.shape[0], 10)
        self.assertTrue((sim_expr.values >= 0).all())
        self.assertIsInstance(sigs, dict)
        for k, v in sigs.items():
            self.assertIn("targets", v)
            self.assertIn("effects", v)



    






    


    




    
    

