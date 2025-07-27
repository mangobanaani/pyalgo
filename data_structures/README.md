# Data Structures

Fundamental data structure implementations with efficient operations and comprehensive analysis.

## Data Structures Included

### Stack (`Stack.py`)
- **Type**: LIFO (Last In, First Out) data structure
- **Operations**: push, pop, top, is_empty, size
- **Time Complexity**: O(1) for all operations
- **Space Complexity**: O(n) where n is number of elements
- **Implementation**: Dynamic array-based
- **Use Cases**: Function call management, expression evaluation, undo operations, backtracking algorithms

### Queue (`Queue.py`)
- **Type**: FIFO (First In, First Out) data structure
- **Operations**: enqueue, dequeue, front, is_empty, size
- **Time Complexity**: O(1) amortized for all operations
- **Space Complexity**: O(n) where n is number of elements
- **Implementation**: Circular array with dynamic resizing
- **Use Cases**: Task scheduling, breadth-first search, handling requests in web servers

### Linked List (`linked_list.py`)
- **Type**: Linear data structure with dynamic memory allocation
- **Operations**: append, prepend, insert, delete, find, reverse, size
- **Time Complexity**: 
  - Insert/Delete at head: O(1)
  - Insert/Delete at arbitrary position: O(n)
  - Search: O(n)
- **Space Complexity**: O(n) with additional pointer overhead
- **Implementation**: Singly linked list with node objects
- **Use Cases**: Dynamic memory allocation, implementing other data structures

### Binary Search Tree (`binary_search_tree.py`)
- **Type**: Hierarchical data structure maintaining sorted order
- **Operations**: insert, search, delete, traversals (inorder, preorder, postorder)
- **Time Complexity**:
  - Average case: O(log n) for insert, search, delete
  - Worst case: O(n) for unbalanced tree
- **Space Complexity**: O(n) for storage, O(h) for recursion where h is height
- **Implementation**: Node-based with left and right children
- **Use Cases**: Efficient searching and sorting, database indexing, expression parsing

## Usage Examples

### Stack Operations
```python
from data_structures.Stack import Stack

# Create and use stack
stack = Stack()

# Push elements
stack.push(1)
stack.push(2)
stack.push(3)

print(f"Stack size: {stack.size()}")  # 3
print(f"Top element: {stack.top()}")  # 3

# Pop elements
while not stack.is_empty():
    print(f"Popped: {stack.pop()}")  # 3, 2, 1

# Stack-based expression evaluation
def evaluate_postfix(expression):
    stack = Stack()
    for token in expression.split():
        if token.isdigit():
            stack.push(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.push(a + b)
            elif token == '*':
                stack.push(a * b)
    return stack.pop()

result = evaluate_postfix("3 4 + 2 *")  # (3 + 4) * 2 = 14
print(f"Postfix result: {result}")
```

### Queue Operations
```python
from data_structures.Queue import Queue

# Create and use queue
queue = Queue()

# Enqueue elements
queue.enqueue("first")
queue.enqueue("second")
queue.enqueue("third")

print(f"Queue size: {queue.size()}")  # 3
print(f"Front element: {queue.front()}")  # "first"

# Dequeue elements
while not queue.is_empty():
    print(f"Dequeued: {queue.dequeue()}")  # first, second, third

# BFS traversal example
def bfs_example():
    queue = Queue()
    queue.enqueue("start")
    visited = set()
    
    while not queue.is_empty():
        current = queue.dequeue()
        if current not in visited:
            visited.add(current)
            print(f"Visiting: {current}")
            # Add neighbors to queue...
```

### Linked List Operations
```python
from data_structures.linked_list import LinkedList

# Create and populate linked list
ll = LinkedList()

# Add elements
ll.append(1)
ll.append(2)
ll.append(3)
ll.prepend(0)

print(f"List size: {ll.size()}")  # 4
print(f"List contents: {ll}")  # 0 -> 1 -> 2 -> 3

# Search and modify
index = ll.find(2)
print(f"Found 2 at index: {index}")  # 2

ll.insert(1, 0.5)  # Insert 0.5 at index 1
ll.delete(3)       # Delete element 3

print(f"After modifications: {ll}")

# Reverse the list
ll.reverse()
print(f"Reversed list: {ll}")
```

### Binary Search Tree Operations
```python
from data_structures.binary_search_tree import BinarySearchTree

# Create and populate BST
bst = BinarySearchTree()

# Insert elements
elements = [5, 3, 7, 2, 4, 6, 8]
for element in elements:
    bst.insert(element)

# Search operations
print(f"Search 4: {bst.search(4)}")  # True
print(f"Search 9: {bst.search(9)}")  # False

# Tree traversals
print(f"Inorder traversal: {bst.inorder_traversal()}")    # [2, 3, 4, 5, 6, 7, 8]
print(f"Preorder traversal: {bst.preorder_traversal()}")  # [5, 3, 2, 4, 7, 6, 8]
print(f"Postorder traversal: {bst.postorder_traversal()}")# [2, 4, 3, 6, 8, 7, 5]

# Delete operations
bst.delete(3)
print(f"After deleting 3: {bst.inorder_traversal()}")

# Tree properties
print(f"Tree height: {bst.height()}")
print(f"Minimum value: {bst.find_min()}")
print(f"Maximum value: {bst.find_max()}")
```

## Performance Analysis

### Time Complexity Comparison

| Operation | Stack | Queue | Linked List | BST (Avg) | BST (Worst) |
|-----------|-------|-------|-------------|-----------|-------------|
| Insert/Add | O(1) | O(1) | O(1) at head | O(log n) | O(n) |
| Delete/Remove | O(1) | O(1) | O(1) at head | O(log n) | O(n) |
| Search | N/A | N/A | O(n) | O(log n) | O(n) |
| Access by Index | N/A | N/A | O(n) | N/A | N/A |

### Space Complexity

| Data Structure | Space Complexity | Additional Overhead |
|----------------|------------------|-------------------|
| Stack | O(n) | Minimal (array-based) |
| Queue | O(n) | Circular array overhead |
| Linked List | O(n) | Pointer storage per node |
| Binary Search Tree | O(n) | Two pointers per node |

## Use Case Guidelines

### Choose Stack When:
- LIFO behavior is required
- Implementing recursive algorithms iteratively
- Parsing expressions or checking balanced parentheses
- Undo/redo functionality
- Backtracking algorithms

### Choose Queue When:
- FIFO behavior is required
- Breadth-first search algorithms
- Task scheduling and buffer management
- Producer-consumer scenarios
- Level-order tree traversal

### Choose Linked List When:
- Frequent insertions/deletions at arbitrary positions
- Size of data structure varies significantly
- Memory allocation needs to be dynamic
- Implementing other data structures (stacks, queues)

### Choose Binary Search Tree When:
- Maintaining sorted data with frequent searches
- Need for efficient insertion and deletion
- Implementing symbol tables or dictionaries
- Range queries on sorted data
- When balanced tree variants aren't needed

## Implementation Details

### Stack Features
- Dynamic resizing for unlimited capacity
- Exception handling for empty stack operations
- Memory-efficient array-based implementation

### Queue Features
- Circular buffer to avoid shifting elements
- Automatic resizing when capacity is exceeded
- Efficient space utilization

### Linked List Features
- Generic implementation supporting any data type
- Iterator support for easy traversal
- Memory allocation only when needed

### Binary Search Tree Features
- Recursive and iterative implementations
- Multiple traversal methods
- Deletion handling for all cases (leaf, one child, two children)
- Parent pointer maintenance for efficient operations

## Advanced Considerations

### Thread Safety
- None of these implementations are thread-safe
- Use appropriate locking mechanisms in concurrent environments
- Consider using concurrent data structures for multi-threaded applications

### Memory Management
- Linked structures may have cache locality issues
- Array-based structures generally have better cache performance
- Consider memory fragmentation with frequent allocations/deallocations

### Optimization Opportunities
- BST can be improved with self-balancing (AVL, Red-Black trees)
- Linked lists can use memory pools for better performance
- Stacks and queues can use specialized memory layouts
