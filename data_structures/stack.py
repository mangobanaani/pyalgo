class Stack:
    """
    Array-based stack implementation
    LIFO (Last In, First Out) data structure
    """
    def __init__(self):
        """Initialize an empty stack"""
        self.data = []

    def __len__(self):
        """Return the number of elements in the stack"""
        return len(self.data)
    
    def len(self):
        """Return the number of elements in the stack (deprecated, use len(stack))"""
        return len(self.data)

    def is_empty(self):
        """Check if the stack is empty"""
        return len(self.data) == 0

    def push(self, e):
        """Add an element to the top of the stack"""
        self.data.append(e)

    def top(self):
        """Return the top element without removing it"""
        if self.is_empty():
            raise Exception('stack is empty')
        return self.data[-1]

    def pop(self) -> object:
        """Remove and return the top element"""
        if self.is_empty():
            raise Exception('stack is empty')
        return self.data.pop()
