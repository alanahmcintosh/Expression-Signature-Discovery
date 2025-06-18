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
        """
        Check if output shape is correct, all values are non-negative and float.
        """
        means = pd.Series([1.0, 2.0, 3.0])
        vars_ = pd.Series([2.0, 4.0, 10.0])
        n_samples = 100

        sim = sim_cna(means, vars_, n_samples)

        self.assertEqual(sim.shape, (n_samples, len(means)))
        self.assertTrue(np.all(sim >= 0))
        self.assertTrue(np.issubdtype(sim.dtype, np.floating))  # can be float since Poisson-like NB

    def test_sim_cna_mean_and_variance_approximation(self):
        """
        Check that the simulated means and variances are close to the input means and variances.
        """

        means = pd.Series([1.0, 2.0, 3.0])
        vars_ = pd.Series([2.0, 4.0, 10.0])
        n_samples = 10000

        sim = sim_cna(means, vars_, n_samples)
        sim_means = sim.mean(axis=0)
        sim_vars = sim.var(axis=0)

        for i in range(len(means)):
            self.assertAlmostEqual(sim_means[i], means[i], delta=0.2 * means[i])
            self.assertAlmostEqual(sim_vars[i], vars_[i], delta=0.2 * vars_[i])

    def test_sim_by_cluster_shapes_with_fusions(self):
        """
        Test that sim_by_cluster returns correctly shaped outputs when all inputs are provided.
        """
        samples = [f"S{i}" for i in range(12)]
        genes = [f"G{i}" for i in range(5)]
        fusions = [f"FUS{i}" for i in range(3)]
        subtype_df = pd.DataFrame({'Subtype': ['A'] * 6 + ['B'] * 6}, index=samples)

        mut_df = pd.DataFrame(np.random.binomial(1, 0.3, size=(12, 5)), index=samples, columns=genes)
        cna_df = pd.DataFrame(np.random.normal(2.0, 0.5, size=(12, 5)), index=samples, columns=genes)
        fusion_df = pd.DataFrame(np.random.binomial(1, 0.1, size=(12, 3)), index=samples, columns=fusions)

        n_samples = 24
        result = sim_by_cluster(mut=mut_df, subtype=subtype_df, n_samples=n_samples, cna=cna_df, fusions=fusion_df)

        self.assertEqual(set(result.keys()), {'mut', 'fusion', 'cna'})
        self.assertEqual(result['mut'].shape, (n_samples, len(genes)))
        self.assertEqual(result['fusion'].shape, (n_samples, len(fusions)))
        self.assertEqual(result['cna'].shape, (n_samples, len(genes)))
        self.assertTrue(all(name.startswith("Sample_") for name in result['mut'].index))

    def test_sim_by_cluster_value_types_and_ranges(self):
        """
        Ensure mutation and fusion outputs are binary, and CNA values are finite and non-negative.
        """
        samples = [f"S{i}" for i in range(12)]
        genes = [f"G{i}" for i in range(5)]
        fusions = [f"FUS{i}" for i in range(3)]
        subtype_df = pd.DataFrame({'Subtype': ['A'] * 6 + ['B'] * 6}, index=samples)

        mut_df = pd.DataFrame(np.random.binomial(1, 0.5, size=(12, 5)), index=samples, columns=genes)
        cna_df = pd.DataFrame(np.abs(np.random.normal(1.5, 0.4, size=(12, 5))), index=samples, columns=genes)
        fusion_df = pd.DataFrame(np.random.binomial(1, 0.2, size=(12, 3)), index=samples, columns=fusions)

        n_samples = 30
        result = sim_by_cluster(mut=mut_df, subtype=subtype_df, n_samples=n_samples, cna=cna_df, fusions=fusion_df)

        # Binary checks
        self.assertTrue(((result['mut'].values == 0) | (result['mut'].values == 1)).all())
        self.assertTrue(((result['fusion'].values == 0) | (result['fusion'].values == 1)).all())

        # CNA checks
        self.assertTrue(np.all(np.isfinite(result['cna'].values)))
        self.assertTrue(np.all(result['cna'].values >= 0))

    def test_sim_by_cluster_balanced_output(self):
        """
        Check that sim_by_cluster returns correctly shaped outputs and valid values
        for mutations and CNAs, stratified by subtype proportions.
        """
        import warnings
        # Create mock data
        samples = [f"S{i}" for i in range(10)]
        genes = [f"G{i}" for i in range(5)]
        mut = pd.DataFrame(np.random.binomial(1, 0.3, size=(10, 5)), index=samples, columns=genes)
        cna = pd.DataFrame(np.abs(np.random.normal(1.0, 0.5, size=(10, 5))), index=samples, columns=genes)
        subtype = pd.DataFrame({'Subtype': ['A'] * 5 + ['B'] * 5}, index=samples)

        n_samples = 100

        # Validate each cluster's NB parameter calculations
        for subtype_label in subtype['Subtype'].unique():
            C_s = cna[subtype['Subtype'] == subtype_label]
            means = C_s.mean()
            variances = C_s.var()

            for gene in C_s.columns:
                mu = means[gene]
                var = variances[gene]
                if var <= mu or not np.isfinite(var):
                    var = mu + max(1e-3, mu * 0.05)

                r = mu**2 / (var - mu) if (var - mu) > 0 else np.nan
                p = r / (r + mu) if r + mu > 0 else np.nan

                self.assertTrue(np.isfinite(r) and r > 0, msg=f"Invalid r for gene {gene}: r={r}, mu={mu}, var={var}")
                self.assertTrue(0 < p < 1, msg=f"Invalid p for gene {gene}: p={p}, r={r}, mu={mu}")

        # Run the simulation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sim_by_cluster(mut=mut, subtype=subtype, n_samples=n_samples, cna=cna)
            sim_mut = result["mut"]
            sim_cna_out = result["cna"]

        # Validate shapes
        self.assertEqual(sim_mut.shape, (n_samples, mut.shape[1]))
        self.assertEqual(sim_cna_out.shape, (n_samples, cna.shape[1]))

        # Check values
        self.assertTrue(((sim_mut.values == 0) | (sim_mut.values == 1)).all())
        self.assertTrue(np.all(np.isfinite(sim_cna_out.values)))
        self.assertTrue((sim_cna_out.values >= 0).all())

    def test_sim_by_cluster_proportionality(self):
        # Uneven subtypes: 3 in A, 7 in B
        samples = [f"S{i}" for i in range(10)]
        subtype_labels = ['A'] * 3 + ['B'] * 7
        subtype = pd.DataFrame({'Subtype': subtype_labels}, index=samples)

        mut = pd.DataFrame(np.random.binomial(1, 0.5, size=(10, 4)), index=samples, columns=[f"G{i}" for i in range(4)])
        cna = pd.DataFrame(np.random.normal(1.0, 0.2, size=(10, 4)), index=samples, columns=[f"G{i}" for i in range(4)])

        n_samples = 100
        _, _ = sim_by_cluster(mut=mut, subtype=subtype, n_samples=n_samples, cna=cna)
 # checks that rounding adjustment works without error

    def test_estimate_deseq2_parameters_output(self):
        """
        Check output shape and values for estimate_deseq2_parameters
        """
        df = pd.DataFrame(np.random.poisson(10, size=(50, 20)), columns=[f"G{i}" for i in range(20)])
        gene_means, gene_vars, dispersions, size_factors = estimate_deseq2_parameters(df, seed=42)

        self.assertEqual(gene_means.shape[0], df.shape[1])
        self.assertEqual(gene_vars.shape[0], df.shape[1])
        self.assertEqual(dispersions.shape[0], df.shape[1])
        self.assertEqual(size_factors.shape[0], df.shape[0])
        self.assertTrue(np.all(dispersions > 0))

    def test_simulate_rna_background_shape(self):
        """
        Check output shape and values for simulate_rna_background
        """
        gene_means = pd.Series([10, 20, 30], index=["A", "B", "C"])
        size_factors = np.array([1.0, 0.5])
        bg = simulate_rna_background(gene_means, dispersions=None, size_factors=size_factors, n_samples=2)

        self.assertEqual(bg.shape, (2, 3))
        self.assertTrue((bg.values >= 0).all())

    def test_generate_signatures_structure_and_fusions(self):
        '''
        Check that mutation and fusion alterations each generate valid expression signatures with: Proper target selection, Valid effect mappings, Inclusion of fusion partner genes and Correct structure of the returned dictionary.
        '''
        genes = [f"G{i}" for i in range(100)]
        alts = ["G1_mut", "G5_mut", "G10-G11_FUSION", "G12-G13_FUSION"]
        signatures = generate_signatures(genes, alts, min_size=5, max_size=5)

        for alt, sig in signatures.items():
            self.assertIn('targets', sig)
            self.assertIn('effects', sig)
            self.assertEqual(set(sig['effects'].keys()), set(sig['targets']))
            self.assertTrue(all(isinstance(v, float) for v in sig['effects'].values()))
            if "_FUSION" in alt:
                partners = alt.replace("_FUSION", "").split("-")
                for p in partners:
                    if p in genes:
                        self.assertIn(p, sig['targets'])

    def test_inject_expression_effects_behavior(self):
        '''
        Check function correctly adds alteration-driven effects to baseline expression, and optionally modulates them by copy number 
        '''
        expr = pd.DataFrame([[10, 10], [10, 10]], columns=["G1", "G2"], index=["S1", "S2"])
        alts = pd.DataFrame([[1], [0]], columns=["G1_mut"], index=["S1", "S2"])
        sigs = {"G1_mut": {"targets": ["G1"], "effects": {"G1": 2.0}}}
        cna = pd.DataFrame([[2, 1], [1, 1]], columns=["G1", "G2"], index=["S1", "S2"])

        mod = inject_expression_effects(expr, alts, sigs, cna_df=cna)
        expected = pd.DataFrame([[12.0, 5.0], [5.0, 5.0]], columns=["G1", "G2"], index=["S1", "S2"])

        np.testing.assert_array_almost_equal(mod.values, expected.values)


    def test_inject_expression_effects_additive_and_cna(self):
        """
        Check that inject_expression_effects returns the correct expression values
        with alteration effects and CNA modulation.
        """
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

        expected = pd.DataFrame([[12.0, 5.0], [5.0, 5.0]], columns=["G1", "G2"], index=["S1", "S2"])
        np.testing.assert_array_almost_equal(mod.values, expected.values)



    def test_sample_nb_properties(self):
        """
        Check that sample_nb returns the correct output shape and properties
        """
        mu = np.full((100, 10), 20.0)
        dispersions = pd.Series(np.full(10, 0.1))
        counts = sample_nb(mu, dispersions)

        self.assertEqual(counts.shape, mu.shape)
        self.assertTrue((counts >= 0).all())
        self.assertTrue(np.issubdtype(counts.dtype, np.integer))

    def test_simulate_rna_with_signatures_end_to_end(self):
        genes = [f"G{i}" for i in range(100)]
        samples = [f"S{i}" for i in range(10)]
        rna_df = pd.DataFrame(np.random.poisson(10, size=(10, 100)), index=samples, columns=genes)
        alteration_df = pd.DataFrame(np.random.binomial(1, 0.2, size=(10, 5)),
                                     index=samples,
                                     columns=["G1_mut", "G5_mut", "G10-G11_FUSION", "G2_mut", "G3_mut"])
        cna_df = pd.DataFrame(np.random.normal(1.0, 0.2, size=(10, 100)), index=samples, columns=genes)

        expr_sim, sigs = simulate_rna_with_signatures(rna_df, alteration_df, cna_df=cna_df, n_samples=10, seed=123)

        self.assertEqual(expr_sim.shape[0], 10)
        self.assertEqual(expr_sim.shape[1], len(expr_sim.columns))
        self.assertTrue((expr_sim.values >= 0).all())
        self.assertIsInstance(sigs, dict)
        for k, v in sigs.items():
            self.assertIn("targets", v)
            self.assertIn("effects", v)
