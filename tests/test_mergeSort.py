import unittest
from unittest import TestCase

from MergeSort import MergeSort


class TestMergeSort(TestCase):
    def test_merge_sort(self):
        stuff = [3, 2, 1]
        MergeSort.merge_sort(stuff)
        expected = [1, 2, 3]
        self.assertListEqual(stuff, expected, "sort not as expected")


if __name__ == "__main__":
    unittest.main()
