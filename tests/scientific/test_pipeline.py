"""Tests for scientific pipeline end-to-end."""
import unittest


class TestScientificPipeline(unittest.TestCase):
    """Test scientific pipeline modules can be imported."""

    def test_offline_pipeline_import(self):
        from pipelines.scientific.offline_pipeline import OfflinePipeline
        self.assertIsNotNone(OfflinePipeline)

    def test_online_pipeline_import(self):
        from pipelines.scientific.online_pipeline import OnlinePipeline
        self.assertIsNotNone(OnlinePipeline)


if __name__ == "__main__":
    unittest.main()
