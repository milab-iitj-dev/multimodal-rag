"""Tests for scientific embedding models."""
import unittest


class TestColPaliEmbedder(unittest.TestCase):
    """Test ColPali multi-vector embedder."""

    def test_import(self):
        from src.domains.scientific.embeddings.colpali_embedder import ColPaliEmbedder
        self.assertIsNotNone(ColPaliEmbedder)


class TestSciNCLEmbedder(unittest.TestCase):
    """Test SciNCL text embedder."""

    def test_import(self):
        from src.domains.scientific.embeddings.scincl_embedder import SciNCLEmbedder
        self.assertIsNotNone(SciNCLEmbedder)


if __name__ == "__main__":
    unittest.main()
