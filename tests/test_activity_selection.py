import unittest
from greedy.activity_selection import activity_selection

class TestActivitySelection(unittest.TestCase):
    def test_non_overlapping_activities(self):
        activities = [(1, 3), (2, 5), (4, 6), (6, 7), (5, 8), (8, 9)]
        selected = activity_selection(activities)
        self.assertEqual(selected, [(1, 3), (4, 6), (6, 7), (8, 9)])

    def test_no_activities(self):
        activities = []
        selected = activity_selection(activities)
        self.assertEqual(selected, [])

    def test_single_activity(self):
        activities = [(1, 2)]
        selected = activity_selection(activities)
        self.assertEqual(selected, [(1, 2)])

if __name__ == "__main__":
    unittest.main()
