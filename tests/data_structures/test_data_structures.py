import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data_structures.linked_list import LinkedList
from data_structures.binary_search_tree import BinarySearchTree


class TestLinkedList(unittest.TestCase):
    """Test suite for LinkedList"""
    
    def setUp(self):
        """Set up a fresh linked list for each test"""
        self.ll = LinkedList()
    
    def test_empty_list(self):
        """Test operations on empty list"""
        self.assertTrue(self.ll.is_empty())
        self.assertEqual(self.ll.length(), 0)
        self.assertEqual(self.ll.display(), [])
    
    def test_append(self):
        """Test appending elements"""
        self.ll.append(1)
        self.ll.append(2)
        self.ll.append(3)
        
        self.assertFalse(self.ll.is_empty())
        self.assertEqual(self.ll.length(), 3)
        self.assertEqual(self.ll.display(), [1, 2, 3])
    
    def test_prepend(self):
        """Test prepending elements"""
        self.ll.prepend(1)
        self.ll.prepend(2)
        self.ll.prepend(3)
        
        self.assertEqual(self.ll.display(), [3, 2, 1])
        self.assertEqual(self.ll.length(), 3)
    
    def test_find(self):
        """Test finding elements"""
        self.ll.append(1)
        self.ll.append(2)
        self.ll.append(3)
        
        self.assertTrue(self.ll.find(2))
        self.assertFalse(self.ll.find(5))
    
    def test_delete(self):
        """Test deleting elements"""
        self.ll.append(1)
        self.ll.append(2)
        self.ll.append(3)
        
        # Delete middle element
        self.assertTrue(self.ll.delete(2))
        self.assertEqual(self.ll.display(), [1, 3])
        self.assertEqual(self.ll.length(), 2)
        
        # Delete first element
        self.assertTrue(self.ll.delete(1))
        self.assertEqual(self.ll.display(), [3])
        
        # Try to delete non-existing element
        self.assertFalse(self.ll.delete(5))
    
    def test_reverse(self):
        """Test reversing the list"""
        self.ll.append(1)
        self.ll.append(2)
        self.ll.append(3)
        self.ll.append(4)
        
        self.ll.reverse()
        self.assertEqual(self.ll.display(), [4, 3, 2, 1])


class TestBinarySearchTree(unittest.TestCase):
    """Test suite for BinarySearchTree"""
    
    def setUp(self):
        """Set up a fresh BST for each test"""
        self.bst = BinarySearchTree()
    
    def test_insert_and_search(self):
        """Test insertion and searching"""
        self.bst.insert(5)
        self.bst.insert(3)
        self.bst.insert(7)
        self.bst.insert(1)
        self.bst.insert(9)
        
        self.assertTrue(self.bst.search(5))
        self.assertTrue(self.bst.search(3))
        self.assertTrue(self.bst.search(7))
        self.assertTrue(self.bst.search(1))
        self.assertTrue(self.bst.search(9))
        self.assertFalse(self.bst.search(10))
    
    def test_traversals(self):
        """Test different tree traversals"""
        self.bst.insert(5)
        self.bst.insert(3)
        self.bst.insert(7)
        self.bst.insert(1)
        self.bst.insert(9)
        
        # Inorder should give sorted order
        self.assertEqual(self.bst.inorder_traversal(), [1, 3, 5, 7, 9])
        
        # Preorder should start with root
        preorder = self.bst.preorder_traversal()
        self.assertEqual(preorder[0], 5)
        
        # Postorder should end with root
        postorder = self.bst.postorder_traversal()
        self.assertEqual(postorder[-1], 5)
    
    def test_delete(self):
        """Test deletion of nodes"""
        self.bst.insert(5)
        self.bst.insert(3)
        self.bst.insert(7)
        self.bst.insert(1)
        self.bst.insert(9)
        
        # Delete leaf node
        self.bst.delete(1)
        self.assertFalse(self.bst.search(1))
        self.assertEqual(self.bst.inorder_traversal(), [3, 5, 7, 9])
        
        # Delete node with one child
        self.bst.delete(7)
        self.assertFalse(self.bst.search(7))
        self.assertEqual(self.bst.inorder_traversal(), [3, 5, 9])


if __name__ == "__main__":
    unittest.main()
