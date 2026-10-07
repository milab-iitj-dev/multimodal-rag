"""Tests for scientific embedding models."""
import unittest


class TestColPaliEmbedder(unittest.TestCase):
    """Test ColPali multi-vector embedder."""

    def test_import(self):
        try:
            from src.domains.scientific.embeddings.colpali_embedder import ColPaliEmbedder
            self.assertIsNotNone(ColPaliEmbedder)
        except ImportError as e:
            raise unittest.SkipTest(f"ColPaliEmbedder dependency not installed: {e}")


class TestSciNCLEmbedder(unittest.TestCase):
    """Test SciNCL text embedder."""

    def test_import(self):
        try:
            from src.domains.scientific.embeddings.scincl_embedder import SciNCLEmbedder
            self.assertIsNotNone(SciNCLEmbedder)
        except ImportError as e:
            raise unittest.SkipTest(f"SciNCLEmbedder dependency not installed: {e}")


if __name__ == "__main__":
    unittest.main()
