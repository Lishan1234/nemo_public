import tensorflow as tf
import ops
from abc import abstractmethod

class MnasNet(tf.keras.Model):
    def __init__(self):
        super(MnasNet, self).__init__()

    @abstractmethod
    def get_name(self):
        pass
