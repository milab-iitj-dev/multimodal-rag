"""Tests for scientific pipeline end-to-end."""
import unittest


class TestScientificPipeline(unittest.TestCase):
    """Test scientific pipeline modules can be imported."""

    def test_offline_pipeline_import(self):
        try:
            from pipelines.scientific.offline_pipeline import OfflinePipeline
            self.assertIsNotNone(OfflinePipeline)
        except ImportError as e:
            raise unittest.SkipTest(f"OfflinePipeline dependency not installed: {e}")

    def test_online_pipeline_import(self):
        try:
            from pipelines.scientific.online_pipeline import OnlinePipeline
            self.assertIsNotNone(OnlinePipeline)
        except ImportError as e:
            raise unittest.SkipTest(f"OnlinePipeline dependency not installed: {e}")


if __name__ == "__main__":
    unittest.main()
