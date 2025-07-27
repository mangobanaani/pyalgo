class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

    def value_per_weight(self):
        return self.value / self.weight

def fractional_knapsack(capacity, items):
    """
    Solve the fractional knapsack problem.

    :param capacity: Maximum weight capacity of the knapsack.
    :param items: List of Item objects with value and weight.
    :return: Maximum value that can be obtained.
    """
    # Sort items by value-to-weight ratio in descending order
    items.sort(key=lambda item: item.value_per_weight(), reverse=True)

    total_value = 0.0
    for item in items:
        if capacity >= item.weight:
            # Take the whole item
            capacity -= item.weight
            total_value += item.value
        else:
            # Take the fraction of the item that fits
            total_value += item.value_per_weight() * capacity
            break

    return total_value
