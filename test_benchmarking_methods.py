import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from Benchmarking_functions import *
class TestSignatureMethods(unittest.TestCase):

    def setUp(self):
        # Simulate mutation (binary) and expression (count) data
        X_data, _ = make_classification(n_samples=50, n_features=10, n_informative=3, random_state=42)
        self.X = pd.DataFrame((X_data > 0).astype(int), columns=[f"G{i}_mut" for i in range(10)])
        self.Y = pd.DataFrame(np.random.poisson(10, size=(50, 100)), columns=[f"Gene{i}" for i in range(100)])
        self.gof = self.X.columns[0]
        self.global_results = {
            'Deconfounder': pd.DataFrame(
                np.random.randn(100, 1),
                index=self.Y.columns,
                columns=[self.gof]
            )
        }

    def test_get_lasso_signature(self):
        # Check that Lasso returns a valid list of gene names
        sig = get_lasso_signature(self.Y, self.X[self.gof])
        self.assertIsInstance(sig, list)

    def test_get_elasticnet_signature(self):
        # Check that ElasticNet returns a valid list of gene names
        sig = get_elasticnet_signature(self.Y, self.X[self.gof])
        self.assertIsInstance(sig, list)

    def test_get_svm_signature(self):
        # Check that SVM returns a valid list of gene names
        sig = get_svm_signature(self.Y, self.X[self.gof])
        self.assertIsInstance(sig, list)

    def test_get_ridgereg_signature(self):
        # Check that Ridge Regression returns a valid list of gene names
        sig = get_ridgereg_signature(self.Y, self.X[self.gof])
        self.assertIsInstance(sig, list)

    def test_get_rf_signature(self):
        # Check that Random Forest returns a valid list of gene names
        sig = get_rf_signature(self.Y, self.X[self.gof])
        self.assertIsInstance(sig, list)

    def test_get_deconfounder_signature(self):
        # Check that the Deconfounder returns a valid gene signature from precomputed results
        sig = get_deconfounder_signature(self.gof, self.global_results)
        self.assertIsInstance(sig, list)

    def test_get_deseq2_signature(self):
        # Check that DESeq2 returns a valid list of differentially expressed genes
        counts_df = pd.DataFrame(np.random.poisson(10, size=(20, 50)), columns=[f"Gene{i}" for i in range(50)])
        X = pd.DataFrame({'GOF1': [0]*10 + [1]*10}, index=counts_df.index)
        sig = get_deseq2_signature(X, counts_df, 'GOF1')
        self.assertIsInstance(sig, list)
        self.assertTrue(all(isinstance(g, str) for g in sig))

    def test_create_supervised_signatures(self):
        # Check that create_supervised_signatures returns all expected methods including DESeq2
        sigs = create_supervised_signatures(self.X, self.Y, self.gof, self.global_results)
        self.assertIn("Random Forest", sigs)
        self.assertIn("Lasso", sigs)
        self.assertIn("ElasticNet", sigs)
        self.assertIn("SVM", sigs)
        self.assertIn("Logistic Regression", sigs)
        self.assertIn("Deconfounder", sigs)
        self.assertIn("DESeq2", sigs)

    def test_create_signatures_kmeans(self):
        # Check that K-means unsupervised signature generation returns a valid structure
        sigs = create_signatures_kmeans(self.X, self.Y, method_name='K-means')
        self.assertIsInstance(sigs, dict)
        for mutation, genes in sigs.get('K-means', {}).items():
            self.assertIsInstance(genes, list)

    def test_create_kmeans_nmf_signature(self):
        # Check that NMF-KMeans returns a valid dictionary of gene signatures
        sigs = create_kmeans_nmf_signature(self.Y, self.X, method_name='NMF-KMeans', n_components=5)
        self.assertIsInstance(sigs, dict)
        for mutation, genes in sigs.get('NMF-KMeans', {}).items():
            self.assertIsInstance(genes, list)

    def test_create_unsupervised_signatures(self):
        # Check that wrapper returns combined unsupervised signature dictionary
        sigs = create_unsupervised_signatures(self.X, self.Y)
        self.assertIsInstance(sigs, dict)

unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestSignatureMethods))