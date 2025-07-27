"""
Red-Black Tree Implementation

Self-balancing binary search tree with guaranteed O(log n) operations.
Each node has a color (red or black) with specific balancing rules.
"""

from typing import Optional, Any, List, Iterator
from enum import Enum


class Color(Enum):
    """Node colors for red-black tree."""
    RED = "RED"
    BLACK = "BLACK"


class RBNode:
    """Node in a red-black tree."""
    
    def __init__(self, key: Any, value: Any = None, color: Color = Color.RED):
        self.key = key
        self.value = value
        self.color = color
        self.left: Optional['RBNode'] = None
        self.right: Optional['RBNode'] = None
        self.parent: Optional['RBNode'] = None
    
    def is_red(self) -> bool:
        """Check if node is red."""
        return self.color == Color.RED
    
    def is_black(self) -> bool:
        """Check if node is black."""
        return self.color == Color.BLACK
    
    def __str__(self) -> str:
        color_str = "R" if self.is_red() else "B"
        return f"{self.key}({color_str})"


class RedBlackTree:
    """
    Red-Black Tree implementation.
    
    Properties:
    1. Every node is either red or black
    2. Root is always black
    3. All leaves (NIL) are black
    4. Red nodes have only black children
    5. All paths from node to leaves have same number of black nodes
    """
    
    def __init__(self):
        # Sentinel NIL node (always black)
        self.NIL = RBNode(None, None, Color.BLACK)
        self.root = self.NIL
        self._size = 0
    
    def insert(self, key: Any, value: Any = None) -> None:
        """
        Insert key-value pair into tree.
        
        Time Complexity: O(log n)
        
        Args:
            key: Key to insert
            value: Optional value associated with key
        """
        new_node = RBNode(key, value, Color.RED)
        new_node.left = self.NIL
        new_node.right = self.NIL
        
        parent = None
        current = self.root
        
        # Find insertion position
        while current != self.NIL:
            parent = current
            if new_node.key < current.key:
                current = current.left
            elif new_node.key > current.key:
                current = current.right
            else:
                # Key already exists, update value
                current.value = value
                return
        
        new_node.parent = parent
        
        if parent is None:
            self.root = new_node
        elif new_node.key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        
        self._size += 1
        
        # Fix red-black properties
        self._insert_fixup(new_node)
    
    def _insert_fixup(self, node: RBNode) -> None:
        """Fix red-black tree properties after insertion."""
        while node.parent and node.parent.is_red():
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                
                if uncle.is_red():
                    # Case 1: Uncle is red
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Node is right child
                        node = node.parent
                        self._left_rotate(node)
                    
                    # Case 3: Node is left child
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                
                if uncle.is_red():
                    # Case 1: Uncle is red (mirrored)
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        # Case 2: Node is left child (mirrored)
                        node = node.parent
                        self._right_rotate(node)
                    
                    # Case 3: Node is right child (mirrored)
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._left_rotate(node.parent.parent)
        
        self.root.color = Color.BLACK
    
    def delete(self, key: Any) -> bool:
        """
        Delete node with given key.
        
        Time Complexity: O(log n)
        
        Args:
            key: Key to delete
            
        Returns:
            True if key was found and deleted, False otherwise
        """
        node = self._search_node(key)
        if node == self.NIL:
            return False
        
        self._delete_node(node)
        self._size -= 1
        return True
    
    def _delete_node(self, node: RBNode) -> None:
        """Delete given node from tree."""
        original_color = node.color
        
        if node.left == self.NIL:
            replacement = node.right
            self._transplant(node, node.right)
        elif node.right == self.NIL:
            replacement = node.left
            self._transplant(node, node.left)
        else:
            # Node has two children
            successor = self._minimum(node.right)
            original_color = successor.color
            replacement = successor.right
            
            if successor.parent == node:
                replacement.parent = successor
            else:
                self._transplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor
            
            self._transplant(node, successor)
            successor.left = node.left
            successor.left.parent = successor
            successor.color = node.color
        
        if original_color == Color.BLACK:
            self._delete_fixup(replacement)
    
    def _delete_fixup(self, node: RBNode) -> None:
        """Fix red-black tree properties after deletion."""
        while node != self.root and node.is_black():
            if node == node.parent.left:
                sibling = node.parent.right
                
                if sibling.is_red():
                    # Case 1: Sibling is red
                    sibling.color = Color.BLACK
                    node.parent.color = Color.RED
                    self._left_rotate(node.parent)
                    sibling = node.parent.right
                
                if sibling.left.is_black() and sibling.right.is_black():
                    # Case 2: Both sibling's children are black
                    sibling.color = Color.RED
                    node = node.parent
                else:
                    if sibling.right.is_black():
                        # Case 3: Sibling's right child is black
                        sibling.left.color = Color.BLACK
                        sibling.color = Color.RED
                        self._right_rotate(sibling)
                        sibling = node.parent.right
                    
                    # Case 4: Sibling's right child is red
                    sibling.color = node.parent.color
                    node.parent.color = Color.BLACK
                    sibling.right.color = Color.BLACK
                    self._left_rotate(node.parent)
                    node = self.root
            else:
                # Mirrored cases
                sibling = node.parent.left
                
                if sibling.is_red():
                    sibling.color = Color.BLACK
                    node.parent.color = Color.RED
                    self._right_rotate(node.parent)
                    sibling = node.parent.left
                
                if sibling.right.is_black() and sibling.left.is_black():
                    sibling.color = Color.RED
                    node = node.parent
                else:
                    if sibling.left.is_black():
                        sibling.right.color = Color.BLACK
                        sibling.color = Color.RED
                        self._left_rotate(sibling)
                        sibling = node.parent.left
                    
                    sibling.color = node.parent.color
                    node.parent.color = Color.BLACK
                    sibling.left.color = Color.BLACK
                    self._right_rotate(node.parent)
                    node = self.root
        
        node.color = Color.BLACK
    
    def _transplant(self, old_node: RBNode, new_node: RBNode) -> None:
        """Replace old_node with new_node."""
        if old_node.parent is None:
            self.root = new_node
        elif old_node == old_node.parent.left:
            old_node.parent.left = new_node
        else:
            old_node.parent.right = new_node
        
        new_node.parent = old_node.parent
    
    def _left_rotate(self, node: RBNode) -> None:
        """Perform left rotation around node."""
        right_child = node.right
        node.right = right_child.left
        
        if right_child.left != self.NIL:
            right_child.left.parent = node
        
        right_child.parent = node.parent
        
        if node.parent is None:
            self.root = right_child
        elif node == node.parent.left:
            node.parent.left = right_child
        else:
            node.parent.right = right_child
        
        right_child.left = node
        node.parent = right_child
    
    def _right_rotate(self, node: RBNode) -> None:
        """Perform right rotation around node."""
        left_child = node.left
        node.left = left_child.right
        
        if left_child.right != self.NIL:
            left_child.right.parent = node
        
        left_child.parent = node.parent
        
        if node.parent is None:
            self.root = left_child
        elif node == node.parent.right:
            node.parent.right = left_child
        else:
            node.parent.left = left_child
        
        left_child.right = node
        node.parent = left_child
    
    def search(self, key: Any) -> Optional[Any]:
        """
        Search for key in tree.
        
        Time Complexity: O(log n)
        
        Args:
            key: Key to search for
            
        Returns:
            Value associated with key, or None if not found
        """
        node = self._search_node(key)
        return node.value if node != self.NIL else None
    
    def _search_node(self, key: Any) -> RBNode:
        """Search for node with given key."""
        current = self.root
        
        while current != self.NIL:
            if key == current.key:
                return current
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        
        return self.NIL
    
    def contains(self, key: Any) -> bool:
        """Check if key exists in tree."""
        return self._search_node(key) != self.NIL
    
    def minimum(self) -> Optional[Any]:
        """Return minimum key in tree."""
        if self.root == self.NIL:
            return None
        return self._minimum(self.root).key
    
    def maximum(self) -> Optional[Any]:
        """Return maximum key in tree."""
        if self.root == self.NIL:
            return None
        return self._maximum(self.root).key
    
    def _minimum(self, node: RBNode) -> RBNode:
        """Find minimum node in subtree."""
        while node.left != self.NIL:
            node = node.left
        return node
    
    def _maximum(self, node: RBNode) -> RBNode:
        """Find maximum node in subtree."""
        while node.right != self.NIL:
            node = node.right
        return node
    
    def size(self) -> int:
        """Return number of nodes in tree."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if tree is empty."""
        return self.root == self.NIL
    
    def height(self) -> int:
        """Return height of tree."""
        return self._height(self.root)
    
    def _height(self, node: RBNode) -> int:
        """Calculate height of subtree."""
        if node == self.NIL:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))
    
    def black_height(self) -> int:
        """Return black height of tree."""
        return self._black_height(self.root)
    
    def _black_height(self, node: RBNode) -> int:
        """Calculate black height of subtree."""
        if node == self.NIL:
            return 1
        
        left_bh = self._black_height(node.left)
        if left_bh == 0:
            return 0
        
        right_bh = self._black_height(node.right)
        if right_bh == 0:
            return 0
        
        if left_bh != right_bh:
            return 0  # Violation of red-black property
        
        return left_bh + (1 if node.is_black() else 0)
    
    def inorder_traversal(self) -> List[Any]:
        """Return keys in sorted order."""
        result = []
        self._inorder_helper(self.root, result)
        return result
    
    def _inorder_helper(self, node: RBNode, result: List[Any]) -> None:
        """Helper for inorder traversal."""
        if node != self.NIL:
            self._inorder_helper(node.left, result)
            result.append(node.key)
            self._inorder_helper(node.right, result)
    
    def preorder_traversal(self) -> List[Any]:
        """Return keys in preorder."""
        result = []
        self._preorder_helper(self.root, result)
        return result
    
    def _preorder_helper(self, node: RBNode, result: List[Any]) -> None:
        """Helper for preorder traversal."""
        if node != self.NIL:
            result.append(node.key)
            self._preorder_helper(node.left, result)
            self._preorder_helper(node.right, result)
    
    def validate(self) -> bool:
        """
        Validate red-black tree properties.
        
        Returns:
            True if tree satisfies all red-black properties
        """
        if self.root == self.NIL:
            return True
        
        # Property 2: Root is black
        if self.root.is_red():
            return False
        
        # Check other properties
        return self._validate_helper(self.root) != -1
    
    def _validate_helper(self, node: RBNode) -> int:
        """
        Validate red-black properties recursively.
        
        Returns:
            Black height if valid, -1 if invalid
        """
        if node == self.NIL:
            return 1
        
        # Property 4: Red node has only black children
        if node.is_red():
            if (node.left != self.NIL and node.left.is_red()) or \
               (node.right != self.NIL and node.right.is_red()):
                return -1
        
        left_bh = self._validate_helper(node.left)
        right_bh = self._validate_helper(node.right)
        
        # Property 5: All paths have same black height
        if left_bh == -1 or right_bh == -1 or left_bh != right_bh:
            return -1
        
        return left_bh + (1 if node.is_black() else 0)
    
    def clear(self) -> None:
        """Remove all nodes from tree."""
        self.root = self.NIL
        self._size = 0
    
    def __len__(self) -> int:
        return self._size
    
    def __bool__(self) -> bool:
        return self.root != self.NIL
    
    def __contains__(self, key: Any) -> bool:
        return self.contains(key)
    
    def __getitem__(self, key: Any) -> Any:
        value = self.search(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: Any, value: Any) -> None:
        self.insert(key, value)
    
    def __delitem__(self, key: Any) -> None:
        if not self.delete(key):
            raise KeyError(key)
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over keys in sorted order."""
        return iter(self.inorder_traversal())
    
    def keys(self) -> List[Any]:
        """Return all keys in sorted order."""
        return self.inorder_traversal()
    
    def values(self) -> List[Any]:
        """Return all values in key order."""
        result = []
        self._values_helper(self.root, result)
        return result
    
    def _values_helper(self, node: RBNode, result: List[Any]) -> None:
        """Helper for collecting values."""
        if node != self.NIL:
            self._values_helper(node.left, result)
            result.append(node.value)
            self._values_helper(node.right, result)
    
    def items(self) -> List[tuple]:
        """Return all key-value pairs in key order."""
        result = []
        self._items_helper(self.root, result)
        return result
    
    def _items_helper(self, node: RBNode, result: List[tuple]) -> None:
        """Helper for collecting key-value pairs."""
        if node != self.NIL:
            self._items_helper(node.left, result)
            result.append((node.key, node.value))
            self._items_helper(node.right, result)
