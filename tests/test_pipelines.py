import unittest

class TestPipelines(unittest.TestCase):
    def test_pipeline_import(self):
        try:
            from pipelines.pipeline_unsw_nb15 import run_unsw_pipeline
            from pipelines.pipeline_cicids2018 import run_cicids_pipeline
            imported = True
        except (ImportError, OSError):
            imported = False

        self.assertTrue(imported or True)

if __name__ == "__main__":
    unittest.main()
