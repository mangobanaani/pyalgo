class Node:
    slots = 'element', 'prev', 'next'

    def __init__(self, element: object, prev: object, next: object) -> object:
        self.element = element
        self.prev = prev
        self.next = next


class LinkedList:
    def __init__(self):
        self.size = None
        self.header = Node(None, None, None)
        self.trailer = Node(None, None, None)
        self.header.next = self.trailer
        self.trailer.prev = self.header

    def __sizeof__(self) -> int:
        return super().__sizeof__()

    def is_empty(self) -> object:
        return self.size == 0

    def insert_between(self, e, predecessor, successor):
        newest = Node(e, predecessor, successor)
        predecessor.next = newest
        successor.prev = newest
        self.size += 1
        return newest

    def delete_node(self, node: object) -> object:
        predecessor = node.prev
        successor = node.next
        predecessor.next = successor
        successor.prev = predecessor
        self.size -= 1
        element = node.element
        node.prev = node.next = node.element = None
        return element
