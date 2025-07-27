import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data_structures.stack import Stack


class TestStack(unittest.TestCase):
    def test_pop(self):
        s = Stack()
        self.assertEqual(s.is_empty(),True, "should be empty")
        s.push(1)
        s.push(2)
        self.assertEqual(s.len(),2,"lenght should be now two")
        val = s.pop()
        self.assertEqual(val, 2, "stack should\'ve returned two")
        self.assertEqual(s.len(),1,"lenght should be now one")

