class Stack:
    def __init__(self):
        self.data = []

    def len(self):
        return len(self.data)

    def is_empty(self):
        return len(self.data) == 0

    def push(self, e):
        self.data.append(e)

    def top(self):
        if self.is_empty():
            raise Exception('stack is empty')
        return self.data[-1]

    def pop(self) -> object:
        if self.is_empty():
            raise Exception('stack is empty')
        return self.data.pop()
