from collections import deque, defaultdict

class AhoCorasick:
    def __init__(self):
        self.trie = defaultdict(dict)
        self.output = defaultdict(list)
        self.fail = {}

    def add_pattern(self, pattern):
        node = 0
        for char in pattern:
            if char not in self.trie[node]:
                self.trie[node][char] = len(self.trie)
            node = self.trie[node][char]
        self.output[node].append(pattern)

    def build(self):
        queue = deque()
        for char, node in self.trie[0].items():
            self.fail[node] = 0
            queue.append(node)

        while queue:
            current = queue.popleft()
            for char, next_node in self.trie[current].items():
                queue.append(next_node)
                fail_state = self.fail[current]
                while fail_state and char not in self.trie[fail_state]:
                    fail_state = self.fail[fail_state]
                self.fail[next_node] = self.trie[fail_state].get(char, 0)
                if self.fail[next_node] != next_node:
                    self.output[next_node] = list(set(self.output[next_node]))
                print(f"Node {next_node}: fail -> {self.fail[next_node]}, output -> {self.output[next_node]}")

    def search(self, text):
        node = 0
        results = []
        for i, char in enumerate(text):
            while node and char not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(char, 0)
            for pattern in self.output[node]:
                start_index = i - len(pattern) + 1
                if start_index >= 0:
                    results.append((start_index, pattern))
        return sorted(results, key=lambda x: x[0])
