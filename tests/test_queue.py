import unittest
from unittest import TestCase

from Queue import Queue


class TestQueue(TestCase):
    def test_dequeue(self):
        q=Queue()
        q.enqueue("foobar")
        self.assertEqual(q.len(),1,"lenght should be now one")
        val=q.dequeue()
        self.assertEqual(val,"foobar","dequeued value not as expected")


if __name__ == "__main__":
    unittest.main()