class LinkedListNode:
    """Node class for the linked list"""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """
    Singly Linked List implementation
    """
    def __init__(self):
        self.head = None
        self.size = 0
    
    def is_empty(self):
        """Check if the list is empty"""
        return self.head is None
    
    def __len__(self):
        """Return the length of the list"""
        return self.size
    
    def length(self):
        """Return the length of the list (deprecated, use len(list))"""
        return self.size
    
    def append(self, data):
        """Add an element to the end of the list"""
        new_node = LinkedListNode(data)
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def prepend(self, data):
        """Add an element to the beginning of the list"""
        new_node = LinkedListNode(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def delete(self, data):
        """Delete the first occurrence of data from the list"""
        if self.is_empty():
            raise Exception("List is empty")
        
        if self.head.data == data:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        return False
    
    def find(self, data):
        """Find if data exists in the list"""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False
    
    def display(self):
        """Display all elements in the list"""
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements
    
    def reverse(self):
        """Reverse the linked list"""
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev
