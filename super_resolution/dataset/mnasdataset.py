from abc import ABCMeta, abstractmethod

class MnasDataset(metaclass = ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def create_train_dataset(self):
        pass

    @abstractmethod
    def create_valid_dataset(self):
        pass

    @abstractmethod
    def create_test_dataset(self):
        pass
