class StrategyA:
    def execute(self):
        print('Strategy A')

class StrategyB:
    def execute(self):
        print('strategy B')

class Context:
    def __init__(self, strategy):
        self.strategy = strategy

    def execute_strategy(self):
        self.strategy.execute()

