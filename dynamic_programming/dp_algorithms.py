class DynamicProgramming:
    """
    Collection of Dynamic Programming algorithms
    """
    
    @staticmethod
    def fibonacci_memo(n, memo=None):
        """
        Fibonacci with memoization (top-down DP)
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if memo is None:
            memo = {}
        
        if n in memo:
            return memo[n]
        
        if n <= 1:
            return n
        
        memo[n] = DynamicProgramming.fibonacci_memo(n-1, memo) + DynamicProgramming.fibonacci_memo(n-2, memo)
        return memo[n]
    
    @staticmethod
    def fibonacci_tabulation(n):
        """
        Fibonacci with tabulation (bottom-up DP)
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n <= 1:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]
    
    @staticmethod
    def fibonacci_optimized(n):
        """
        Space-optimized Fibonacci
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n <= 1:
            return n
        
        prev2, prev1 = 0, 1
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    @staticmethod
    def longest_common_subsequence(text1, text2):
        """
        Find the length of the longest common subsequence
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def knapsack_01(weights, values, capacity):
        """
        0/1 Knapsack problem
        Time Complexity: O(n * capacity)
        Space Complexity: O(n * capacity)
        """
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(
                        values[i-1] + dp[i-1][w - weights[i-1]],  # Include item
                        dp[i-1][w]  # Exclude item
                    )
                else:
                    dp[i][w] = dp[i-1][w]
        
        return dp[n][capacity]
    
    @staticmethod
    def coin_change(coins, amount):
        """
        Minimum number of coins to make the amount
        Time Complexity: O(amount * len(coins))
        Space Complexity: O(amount)
        """
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    @staticmethod
    def longest_increasing_subsequence(nums):
        """
        Find the length of the longest increasing subsequence
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    @staticmethod
    def edit_distance(word1, word2):
        """
        Minimum edit distance (Levenshtein distance)
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Delete
                        dp[i][j-1],      # Insert
                        dp[i-1][j-1]     # Replace
                    )
        
        return dp[m][n]
