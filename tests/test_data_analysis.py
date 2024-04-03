import unittest
from unittest.mock import MagicMock, patch

from acels.data_analysis import (
    ModelData,
    extract_data_from_filename,
    process_files_top,
    read_metrics,
    save_top_models,
)


class TestExtractDataFromFilename(unittest.TestCase):
    def test_extract_data_from_filename_valid(self):
        filename = "123_model_non_quant_impl_metrics.txt"
        expected = ("123", "non_quant")
        self.assertEqual(extract_data_from_filename(filename), expected)

    def test_extract_data_from_filename_invalid(self):
        filename = "invalid_filename.txt"
        expected = (None, None)
        self.assertEqual(extract_data_from_filename(filename), expected)


class TestReadMetrics(unittest.TestCase):
    @patch(
        "builtins.open",
        unittest.mock.mock_open(read_data="# MAE: 0.5 mm\n# Average Runtime: 100 us"),
    )
    def test_read_metrics_valid(self):
        file_path = "dummy_path"
        expected = (0.5, 100.0)
        self.assertEqual(read_metrics(file_path), expected)

    @patch("builtins.open", unittest.mock.mock_open(read_data=""))
    def test_read_metrics_invalid(self):
        file_path = "dummy_path"
        expected = (None, None)
        self.assertEqual(read_metrics(file_path), expected)


if __name__ == "__main__":
    unittest.main()
