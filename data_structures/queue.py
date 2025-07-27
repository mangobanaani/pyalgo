class Queue:
    """
    Circular array-based queue implementation
    FIFO (First In, First Out) data structure
    """
    DEFAULT_CAPACITY = 5

    def __init__(self):
        """Initialize an empty queue"""
        self.head = 0
        self.size = 0
        self.data = [None] * Queue.DEFAULT_CAPACITY

    def __len__(self):
        """Return the number of elements in the queue"""
        return self.size
    
    def len(self):
        """Return the number of elements in the queue (deprecated, use len(queue))"""
        return self.size

    def is_empty(self):
        """Check if the queue is empty"""
        return self.size == 0

    def first(self):
        """Return the first element without removing it"""
        if self.is_empty():
            raise Exception('queue is empty')
        return self.data[self.head]

    def dequeue(self):
        """Remove and return the first element"""
        if self.is_empty():
            raise Exception('queue is empty')
        ret = self.data[self.head]
        self.data[self.head] = None
        self.head = (self.head + 1) % len(self.data)
        self.size -= 1
        return ret

    def enqueue(self, e):
        """Add an element to the rear of the queue"""
        if self.size == len(self.data):
            self.resize(2 * len(self.data))
        avail = (self.head + self.size) % len(self.data)
        self.data[avail] = e
        self.size += 1

    def resize(self, cap):
        """Resize the internal array when capacity is exceeded"""
        old = self.data
        self.data = [None]*cap
        walk = self.head
        for k in range(self.size):
            self.data[k] = old[walk]
            walk = (1 + walk) % len(old)
        self.head = 0
