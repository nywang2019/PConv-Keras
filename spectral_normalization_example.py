import tensorflow as tf
from tensorflow.keras import Model, layers
from spectral_normalization import SpectralNormalization


class Cnn(k.Model):
    def __init__(self, name=None):
        super(cnn, self).__init__(name=name)
        self.conv1 = SpectralNormalization(layers.Conv2D(1, (3, 3)))
        self.conv2 = SpectralNormalization(layers.Conv2D(64, (3, 3)))

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training)
        x = self.conv2(x, training)
        return x


cnn = Cnn()
optimizers = tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

####### Training #######
with tf.GradientTape() as g_tape:
    outputs = cnn(inputs, training=True)  # Set training to True is important
    # if training is False, convolution kernel won't be updated by spectral normalization.
    loss = tf.reduce_mean(outputs)
grad = g_tape.gradient(loss, cnn.trainable_variables)
optimizers.apply_gradient(zip(grad, cnn.trainable_varialbles))

####### Inference #######
outputs = cnn(inputs, training=False)