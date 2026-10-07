"""Tests for scientific retrieval modules."""
import unittest


class TestColPaliRetriever(unittest.TestCase):
    """Test ColPali MaxSim retriever."""

    def test_import(self):
        try:
            from src.domains.scientific.retrieval.colpali_retriever import ColPaliRetriever
            self.assertIsNotNone(ColPaliRetriever)
        except ImportError as e:
            raise unittest.SkipTest(f"ColPaliRetriever dependency not installed: {e}")


class TestTextRetriever(unittest.TestCase):
    """Test ChromaDB text retriever."""

    def test_import(self):
        try:
            from src.domains.scientific.retrieval.text_retriever import TextRetriever
            self.assertIsNotNone(TextRetriever)
        except ImportError as e:
            raise unittest.SkipTest(f"TextRetriever dependency not installed: {e}")


class TestFusionRetriever(unittest.TestCase):
    """Test weighted score fusion retriever."""

    def test_import(self):
        try:
            from src.domains.scientific.retrieval.fusion_retriever import FusionRetriever
            self.assertIsNotNone(FusionRetriever)
        except ImportError as e:
            raise unittest.SkipTest(f"FusionRetriever dependency not installed: {e}")


if __name__ == "__main__":
    unittest.main()
