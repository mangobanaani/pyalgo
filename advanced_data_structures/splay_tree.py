"""
Splay Tree Implementation

Self-adjusting binary search tree where recently accessed elements
are moved to the root through splaying operations.
"""

from typing import Optional, Any, List


class SplayNode:
    """Node in a splay tree."""
    
    def __init__(self, key: Any, value: Any = None):
        self.key = key
        self.value = value
        self.left: Optional['SplayNode'] = None
        self.right: Optional['SplayNode'] = None
        self.parent: Optional['SplayNode'] = None
    
    def __str__(self) -> str:
        return f"SplayNode({self.key})"


class SplayTree:
    """
    Splay Tree implementation.
    
    Self-adjusting binary search tree where frequently accessed nodes
    move closer to root, providing good amortized performance for
    sequences of operations with locality of reference.
    """
    
    def __init__(self):
        self.root: Optional[SplayNode] = None
        self._size = 0
    
    def _set_parent(self, child: Optional[SplayNode], parent: Optional[SplayNode]) -> None:
        """Set parent-child relationship."""
        if child is not None:
            child.parent = parent
    
    def _keep_parent(self, node: Optional[SplayNode]) -> None:
        """Update parent pointers for node's children."""
        if node is not None:
            self._set_parent(node.left, node)
            self._set_parent(node.right, node)
    
    def _rotate_right(self, node: SplayNode) -> SplayNode:
        """Perform right rotation."""
        left_child = node.left
        node.left = left_child.right
        left_child.right = node
        
        self._keep_parent(left_child)
        self._keep_parent(node)
        
        return left_child
    
    def _rotate_left(self, node: SplayNode) -> SplayNode:
        """Perform left rotation."""
        right_child = node.right
        node.right = right_child.left
        right_child.left = node
        
        self._keep_parent(right_child)
        self._keep_parent(node)
        
        return right_child
    
    def _splay(self, node: SplayNode) -> SplayNode:
        """
        Splay operation - move node to root.
        
        Uses zig, zig-zig, and zig-zag operations to restructure tree.
        """
        if node.parent is None:
            return node
        
        parent = node.parent
        grandparent = parent.parent
        
        if grandparent is None:
            # Zig step - node is child of root
            if parent.left == node:
                return self._rotate_right(parent)
            else:
                return self._rotate_left(parent)
        
        # Zig-zig and zig-zag steps
        if (grandparent.left == parent) == (parent.left == node):
            # Zig-zig: node and parent are both left or both right children
            if parent.left == node:
                # Both are left children
                self._rotate_right(grandparent)
                node = self._rotate_right(parent)
            else:
                # Both are right children
                self._rotate_left(grandparent)
                node = self._rotate_left(parent)
        else:
            # Zig-zag: node and parent are on opposite sides
            if parent.left == node:
                # Parent is right child, node is left child
                self._rotate_right(parent)
                node = self._rotate_left(grandparent)
            else:
                # Parent is left child, node is right child
                self._rotate_left(parent)
                node = self._rotate_right(grandparent)
        
        # Continue splaying if not at root
        if node.parent is not None:
            node = self._splay(node)
        
        return node
    
    def _find(self, key: Any) -> Optional[SplayNode]:
        """Find node with given key and splay it to root."""
        current = self.root
        last_node = None
        
        while current is not None:
            last_node = current
            if key == current.key:
                self.root = self._splay(current)
                return self.root
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        
        # Key not found, splay last accessed node
        if last_node is not None:
            self.root = self._splay(last_node)
        
        return None
    
    def insert(self, key: Any, value: Any = None) -> None:
        """
        Insert key-value pair into tree.
        
        Time Complexity: O(log n) amortized
        """
        if self.root is None:
            self.root = SplayNode(key, value)
            self._size += 1
            return
        
        # Find insertion position (this also splays)
        node = self._find(key)
        
        if node is not None and node.key == key:
            # Key already exists, update value
            node.value = value
            return
        
        # Create new node and make it root
        new_node = SplayNode(key, value)
        
        if key < self.root.key:
            # New key is smaller than root
            new_node.left = self.root.left
            new_node.right = self.root
            self.root.left = None
            self._set_parent(self.root, new_node)
            self._set_parent(new_node.left, new_node)
        else:
            # New key is larger than root
            new_node.right = self.root.right
            new_node.left = self.root
            self.root.right = None
            self._set_parent(self.root, new_node)
            self._set_parent(new_node.right, new_node)
        
        self.root = new_node
        self._size += 1
    
    def delete(self, key: Any) -> bool:
        """
        Delete node with given key.
        
        Time Complexity: O(log n) amortized
        
        Returns:
            True if key was found and deleted, False otherwise
        """
        node = self._find(key)
        
        if node is None or node.key != key:
            return False
        
        # Node to delete is now at root
        if self.root.left is None:
            self.root = self.root.right
            self._set_parent(self.root, None)
        elif self.root.right is None:
            self.root = self.root.left
            self._set_parent(self.root, None)
        else:
            # Node has both children
            left_subtree = self.root.left
            right_subtree = self.root.right
            
            # Make left subtree the new root
            self.root = left_subtree
            self._set_parent(self.root, None)
            
            # Find maximum in left subtree and splay it
            max_node = self._find_max(left_subtree)
            self.root = self._splay(max_node)
            
            # Attach right subtree
            self.root.right = right_subtree
            self._set_parent(right_subtree, self.root)
        
        self._size -= 1
        return True
    
    def _find_max(self, node: SplayNode) -> SplayNode:
        """Find maximum node in subtree."""
        while node.right is not None:
            node = node.right
        return node
    
    def search(self, key: Any) -> Optional[Any]:
        """
        Search for key in tree.
        
        Time Complexity: O(log n) amortized
        
        Args:
            key: Key to search for
            
        Returns:
            Value associated with key, or None if not found
        """
        node = self._find(key)
        return node.value if node and node.key == key else None
    
    def contains(self, key: Any) -> bool:
        """Check if key exists in tree."""
        return self.search(key) is not None
    
    def minimum(self) -> Optional[Any]:
        """Return minimum key in tree."""
        if self.root is None:
            return None
        
        current = self.root
        while current.left is not None:
            current = current.left
        
        # Splay minimum to root
        self.root = self._splay(current)
        return self.root.key
    
    def maximum(self) -> Optional[Any]:
        """Return maximum key in tree."""
        if self.root is None:
            return None
        
        current = self.root
        while current.right is not None:
            current = current.right
        
        # Splay maximum to root
        self.root = self._splay(current)
        return self.root.key
    
    def size(self) -> int:
        """Return number of nodes in tree."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if tree is empty."""
        return self.root is None
    
    def height(self) -> int:
        """Return height of tree."""
        return self._height(self.root)
    
    def _height(self, node: Optional[SplayNode]) -> int:
        """Calculate height of subtree."""
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))
    
    def inorder_traversal(self) -> List[Any]:
        """Return keys in sorted order."""
        result = []
        self._inorder_helper(self.root, result)
        return result
    
    def _inorder_helper(self, node: Optional[SplayNode], result: List[Any]) -> None:
        """Helper for inorder traversal."""
        if node is not None:
            self._inorder_helper(node.left, result)
            result.append(node.key)
            self._inorder_helper(node.right, result)
    
    def preorder_traversal(self) -> List[Any]:
        """Return keys in preorder."""
        result = []
        self._preorder_helper(self.root, result)
        return result
    
    def _preorder_helper(self, node: Optional[SplayNode], result: List[Any]) -> None:
        """Helper for preorder traversal."""
        if node is not None:
            result.append(node.key)
            self._preorder_helper(node.left, result)
            self._preorder_helper(node.right, result)
    
    def postorder_traversal(self) -> List[Any]:
        """Return keys in postorder."""
        result = []
        self._postorder_helper(self.root, result)
        return result
    
    def _postorder_helper(self, node: Optional[SplayNode], result: List[Any]) -> None:
        """Helper for postorder traversal."""
        if node is not None:
            self._postorder_helper(node.left, result)
            self._postorder_helper(node.right, result)
            result.append(node.key)
    
    def split(self, key: Any) -> tuple['SplayTree', 'SplayTree']:
        """
        Split tree into two trees: one with keys < key, one with keys >= key.
        
        Returns:
            Tuple of (left_tree, right_tree)
        """
        if self.root is None:
            return SplayTree(), SplayTree()
        
        # Find position to split
        self._find(key)
        
        left_tree = SplayTree()
        right_tree = SplayTree()
        
        if self.root.key < key:
            # Split at right subtree
            left_tree.root = self.root
            right_tree.root = self.root.right
            
            if left_tree.root is not None:
                left_tree.root.right = None
                left_tree._set_parent(left_tree.root.right, left_tree.root)
            
            if right_tree.root is not None:
                right_tree._set_parent(right_tree.root, None)
        else:
            # Split at left subtree
            right_tree.root = self.root
            left_tree.root = self.root.left
            
            if right_tree.root is not None:
                right_tree.root.left = None
                right_tree._set_parent(right_tree.root.left, right_tree.root)
            
            if left_tree.root is not None:
                left_tree._set_parent(left_tree.root, None)
        
        # Update sizes (approximate)
        left_tree._size = self._count_nodes(left_tree.root)
        right_tree._size = self._count_nodes(right_tree.root)
        
        self.root = None
        self._size = 0
        
        return left_tree, right_tree
    
    def join(self, other: 'SplayTree') -> 'SplayTree':
        """
        Join this tree with another tree.
        
        Assumes all keys in this tree are smaller than keys in other tree.
        
        Args:
            other: Tree to join with
            
        Returns:
            New joined tree
        """
        if self.root is None:
            return other
        
        if other.root is None:
            return self
        
        # Find maximum in left tree and splay it
        max_node = self._find_max(self.root)
        self.root = self._splay(max_node)
        
        # Attach right tree
        self.root.right = other.root
        self._set_parent(other.root, self.root)
        
        self._size += other._size
        
        # Clear other tree
        other.root = None
        other._size = 0
        
        return self
    
    def _count_nodes(self, node: Optional[SplayNode]) -> int:
        """Count nodes in subtree."""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def clear(self) -> None:
        """Remove all nodes from tree."""
        self.root = None
        self._size = 0
    
    def __len__(self) -> int:
        return self._size
    
    def __bool__(self) -> bool:
        return self.root is not None
    
    def __contains__(self, key: Any) -> bool:
        return self.contains(key)
    
    def __getitem__(self, key: Any) -> Any:
        value = self.search(key)
        if value is None and not self.contains(key):
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: Any, value: Any) -> None:
        self.insert(key, value)
    
    def __delitem__(self, key: Any) -> None:
        if not self.delete(key):
            raise KeyError(key)
    
    def keys(self) -> List[Any]:
        """Return all keys in sorted order."""
        return self.inorder_traversal()
    
    def values(self) -> List[Any]:
        """Return all values in key order."""
        result = []
        self._values_helper(self.root, result)
        return result
    
    def _values_helper(self, node: Optional[SplayNode], result: List[Any]) -> None:
        """Helper for collecting values."""
        if node is not None:
            self._values_helper(node.left, result)
            result.append(node.value)
            self._values_helper(node.right, result)
    
    def items(self) -> List[tuple]:
        """Return all key-value pairs in key order."""
        result = []
        self._items_helper(self.root, result)
        return result
    
    def _items_helper(self, node: Optional[SplayNode], result: List[tuple]) -> None:
        """Helper for collecting key-value pairs."""
        if node is not None:
            self._items_helper(node.left, result)
            result.append((node.key, node.value))
            self._items_helper(node.right, result)
