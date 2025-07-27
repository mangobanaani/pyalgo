"""
AVL Tree Implementation

Self-balancing binary search tree where the heights of the two child subtrees 
of any node differ by at most one. Named after Adelson-Velsky and Landis.
"""

from typing import Optional, List, Tuple


class AVLNode:
    """Node in AVL Tree."""
    
    def __init__(self, key: int, value=None):
        self.key = key
        self.value = value if value is not None else key
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height = 1
        
    def __str__(self):
        return f"AVLNode({self.key})"


class AVLTree:
    """
    AVL Tree - Self-balancing binary search tree.
    
    Maintains balance factor of each node within [-1, 1] through rotations.
    Guarantees O(log n) time complexity for all operations.
    """
    
    def __init__(self):
        self.root: Optional[AVLNode] = None
        self._size = 0
    
    def size(self) -> int:
        """Return number of nodes in tree."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if tree is empty."""
        return self.root is None
    
    def height(self, node: Optional[AVLNode]) -> int:
        """Get height of node. Null nodes have height 0."""
        if not node:
            return 0
        return node.height
    
    def balance_factor(self, node: Optional[AVLNode]) -> int:
        """Calculate balance factor (left height - right height)."""
        if not node:
            return 0
        return self.height(node.left) - self.height(node.right)
    
    def update_height(self, node: AVLNode) -> None:
        """Update height of node based on children heights."""
        node.height = 1 + max(self.height(node.left), self.height(node.right))
    
    def rotate_right(self, y: AVLNode) -> AVLNode:
        """
        Right rotation around node y.
        
        Before:     y               After:      x
                   / \                         / \
                  x   T3                      T1  y
                 / \                             / \
                T1  T2                          T2  T3
        """
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        self.update_height(y)
        self.update_height(x)
        
        return x
    
    def rotate_left(self, x: AVLNode) -> AVLNode:
        """
        Left rotation around node x.
        
        Before:   x                 After:        y
                 / \                             / \
                T1  y                          x   T3
                   / \                        / \
                  T2  T3                     T1  T2
        """
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        self.update_height(x)
        self.update_height(y)
        
        return y
    
    def insert(self, key: int, value=None) -> None:
        """Insert key-value pair into AVL tree."""
        self.root = self._insert(self.root, key, value)
        self._size += 1
    
    def _insert(self, node: Optional[AVLNode], key: int, value=None) -> AVLNode:
        """Recursive helper for insertion."""
        # Step 1: Perform normal BST insertion
        if not node:
            return AVLNode(key, value)
        
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            # Update value for existing key
            node.value = value if value is not None else key
            self._size -= 1  # Don't count as new insertion
            return node
        
        # Step 2: Update height
        self.update_height(node)
        
        # Step 3: Get balance factor
        balance = self.balance_factor(node)
        
        # Step 4: Perform rotations if unbalanced
        # Left Left Case
        if balance > 1 and key < node.left.key:
            return self.rotate_right(node)
        
        # Right Right Case
        if balance < -1 and key > node.right.key:
            return self.rotate_left(node)
        
        # Left Right Case
        if balance > 1 and key > node.left.key:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        
        # Right Left Case
        if balance < -1 and key < node.right.key:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        
        return node
    
    def delete(self, key: int) -> bool:
        """Delete key from AVL tree. Returns True if key was found and deleted."""
        initial_size = self._size
        self.root = self._delete(self.root, key)
        return self._size < initial_size
    
    def _delete(self, node: Optional[AVLNode], key: int) -> Optional[AVLNode]:
        """Recursive helper for deletion."""
        # Step 1: Perform normal BST deletion
        if not node:
            return node
        
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # Node to be deleted found
            self._size -= 1
            
            # Node with only right child or no child
            if not node.left:
                return node.right
            
            # Node with only left child
            if not node.right:
                return node.left
            
            # Node with two children: get inorder successor
            successor = self._min_value_node(node.right)
            
            # Copy successor's data to this node
            node.key = successor.key
            node.value = successor.value
            
            # Delete the successor
            node.right = self._delete(node.right, successor.key)
            self._size += 1  # Compensate for double decrement
        
        # Step 2: Update height
        self.update_height(node)
        
        # Step 3: Get balance factor
        balance = self.balance_factor(node)
        
        # Step 4: Perform rotations if unbalanced
        # Left Left Case
        if balance > 1 and self.balance_factor(node.left) >= 0:
            return self.rotate_right(node)
        
        # Left Right Case
        if balance > 1 and self.balance_factor(node.left) < 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        
        # Right Right Case
        if balance < -1 and self.balance_factor(node.right) <= 0:
            return self.rotate_left(node)
        
        # Right Left Case
        if balance < -1 and self.balance_factor(node.right) > 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        
        return node
    
    def _min_value_node(self, node: AVLNode) -> AVLNode:
        """Find node with minimum key in subtree."""
        while node.left:
            node = node.left
        return node
    
    def search(self, key: int) -> Optional[any]:
        """Search for key in tree. Returns value if found, None otherwise."""
        node = self._search(self.root, key)
        return node.value if node else None
    
    def _search(self, node: Optional[AVLNode], key: int) -> Optional[AVLNode]:
        """Recursive helper for search."""
        if not node or node.key == key:
            return node
        
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)
    
    def contains(self, key: int) -> bool:
        """Check if key exists in tree."""
        return self._search(self.root, key) is not None
    
    def inorder_traversal(self) -> List[int]:
        """Return keys in sorted order."""
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, node: Optional[AVLNode], result: List[int]) -> None:
        """Recursive helper for inorder traversal."""
        if node:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)
    
    def preorder_traversal(self) -> List[int]:
        """Return keys in preorder."""
        result = []
        self._preorder(self.root, result)
        return result
    
    def _preorder(self, node: Optional[AVLNode], result: List[int]) -> None:
        """Recursive helper for preorder traversal."""
        if node:
            result.append(node.key)
            self._preorder(node.left, result)
            self._preorder(node.right, result)
    
    def postorder_traversal(self) -> List[int]:
        """Return keys in postorder."""
        result = []
        self._postorder(self.root, result)
        return result
    
    def _postorder(self, node: Optional[AVLNode], result: List[int]) -> None:
        """Recursive helper for postorder traversal."""
        if node:
            self._postorder(node.left, result)
            self._postorder(node.right, result)
            result.append(node.key)
    
    def get_tree_height(self) -> int:
        """Get height of entire tree."""
        return self.height(self.root)
    
    def is_balanced(self) -> bool:
        """Check if tree is properly balanced."""
        return self._is_balanced(self.root)
    
    def _is_balanced(self, node: Optional[AVLNode]) -> bool:
        """Recursive helper to check balance property."""
        if not node:
            return True
        
        balance = self.balance_factor(node)
        if abs(balance) > 1:
            return False
        
        return self._is_balanced(node.left) and self._is_balanced(node.right)
    
    def get_min(self) -> Optional[int]:
        """Get minimum key in tree."""
        if not self.root:
            return None
        return self._min_value_node(self.root).key
    
    def get_max(self) -> Optional[int]:
        """Get maximum key in tree."""
        if not self.root:
            return None
        node = self.root
        while node.right:
            node = node.right
        return node.key
    
    def range_query(self, low: int, high: int) -> List[int]:
        """Get all keys in range [low, high]."""
        result = []
        self._range_query(self.root, low, high, result)
        return result
    
    def _range_query(self, node: Optional[AVLNode], low: int, high: int, result: List[int]) -> None:
        """Recursive helper for range query."""
        if not node:
            return
        
        if low <= node.key <= high:
            result.append(node.key)
        
        if low < node.key:
            self._range_query(node.left, low, high, result)
        
        if high > node.key:
            self._range_query(node.right, low, high, result)
    
    def print_tree(self) -> None:
        """Print tree structure."""
        if not self.root:
            print("Empty tree")
            return
        
        lines = []
        self._print_tree(self.root, "", True, lines)
        for line in lines:
            print(line)
    
    def _print_tree(self, node: Optional[AVLNode], prefix: str, is_last: bool, lines: List[str]) -> None:
        """Recursive helper for tree printing."""
        if node:
            lines.append(f"{prefix}{'└── ' if is_last else '├── '}{node.key} (h:{node.height}, bf:{self.balance_factor(node)})")
            children = [child for child in [node.left, node.right] if child]
            
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                extension = "    " if is_last else "│   "
                self._print_tree(child, prefix + extension, is_last_child, lines)
