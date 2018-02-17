class Queue:
    DEFAULT_CAPACITY = 5

    def __init__(self):
        self.head = 0
        self.size = 0
        self.data = [None] * Queue.DEFAULT_CAPACITY

    def len(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def first(self):
        if self.is_empty():
            raise Exception('queue is empty')
        return self.data[self.head]

    def dequeue(self):
        if self.is_empty():
            raise Exception('queue is empty')
        ret = self.data[self.head]
        self.data[self.head] = None
        self.head = (self.head + 1) % len(self.data)
        self.size -= 1
        return ret

    def enqueue(self, e):
        if self.size == len(self.data):
            self.resize(2 * len(self.data))
        avail = (self.head + self.size) % len(self.data)
        self.data[avail] = e
        self.size += 1

    def resize(self, cap):
        old = self.data
        self.data = [None]*cap
        walk = self.head
        for k in range(self.size):
            self.data[k] = old[walk]
            walk = (1 + walk) % len(old)
        self.head = 0
