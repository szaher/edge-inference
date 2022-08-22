import abc

class BaseLayer(abc.ABC):
    """Base Layer class.
    Common methods and must implement methods are defined here"""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.input = None
        self.output = None

    @abc.abstractmethod
    def feedforward(self, x):
        pass

    @abc.abstractmethod
    def backpropagation(self, gradient):
        pass


