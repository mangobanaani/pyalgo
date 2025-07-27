import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data_structures.queue import Queue


class TestQueue(unittest.TestCase):
    def test_dequeue(self):
        """Test basic enqueue and dequeue operations"""
        q = Queue()
        q.enqueue("foobar")
        self.assertEqual(q.len(), 1, "length should be now one")
        val = q.dequeue()
        self.assertEqual(val, "foobar", "dequeued value not as expected")
    
    def test_empty_queue(self):
        """Test operations on empty queue"""
        q = Queue()
        self.assertTrue(q.is_empty())
        self.assertEqual(q.len(), 0)
    
    def test_multiple_operations(self):
        """Test multiple enqueue and dequeue operations"""
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        
        self.assertEqual(q.len(), 3)
        self.assertEqual(q.first(), 1)
        self.assertEqual(q.dequeue(), 1)
        self.assertEqual(q.dequeue(), 2)
        self.assertEqual(q.len(), 1)
        self.assertEqual(q.dequeue(), 3)
        self.assertTrue(q.is_empty())


if __name__ == "__main__":
    unittest.main()