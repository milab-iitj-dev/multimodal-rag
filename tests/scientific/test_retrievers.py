"""Tests for scientific retrieval modules."""
import unittest


class TestColPaliRetriever(unittest.TestCase):
    """Test ColPali MaxSim retriever."""

    def test_import(self):
        from src.domains.scientific.retrieval.colpali_retriever import ColPaliRetriever
        self.assertIsNotNone(ColPaliRetriever)


class TestTextRetriever(unittest.TestCase):
    """Test ChromaDB text retriever."""

    def test_import(self):
        from src.domains.scientific.retrieval.text_retriever import TextRetriever
        self.assertIsNotNone(TextRetriever)


class TestFusionRetriever(unittest.TestCase):
    """Test weighted score fusion retriever."""

    def test_import(self):
        from src.domains.scientific.retrieval.fusion_retriever import FusionRetriever
        self.assertIsNotNone(FusionRetriever)


if __name__ == "__main__":
    unittest.main()
