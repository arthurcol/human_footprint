import tensorflow as tf


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, NumFilter: int, BatchNorm: bool, DropOut: float):
        self.batch_norm = BatchNorm

        self.conv2D = tf.keras.layers.Conv2D(
            filters=NumFilter, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )

        self.activ = tf.keras.layers.Activation("relu")
        self.norm_layer = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(DropOut)
        self.pool = tf.keras.layers.MaxPool2D((2, 2))

        super(EncoderBlock, self).__init__()

    def call(self, inputs):
        x = self.conv2D(inputs)
        res = self.activ(x)
        if self.batch_norm:
            res = self.norm_layer(x)
        x = self.pool(res)
        output = self.drop(x)

        return output, res


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, NumFilter: int, BatchNorm: bool, DropOut: float):
        self.batch_norm = BatchNorm

        self.conv2D = tf.keras.layers.Conv2D(
            NumFilter, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )

        self.activ = tf.keras.layers.Activation("relu")
        self.norm_layer = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(DropOut)
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.concat = tf.keras.layers.Concatenate()

        super(DecoderBlock, self).__init__()

    def call(self, inputs, residuals):
        x = self.upsampling(inputs)
        x = self.concat([x, residuals])
        x = self.conv2D(x)
        x = self.activ(x)
        if self.batch_norm:
            x = self.norm_layer(x)
        output = self.drop(x)

        return output


class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, NumFilter: int, BatchNorm: bool, DropOut: float):
        self.batch_norm = BatchNorm

        self.conv2D = tf.keras.layers.Conv2D(
            NumFilter, kernel_size=(3, 3), strides=(1, 1), padding="same"
        )
        self.activ = tf.keras.layers.Activation("relu")
        self.norm_layer = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(DropOut)

        super(BottleneckBlock, self).__init__()

    def call(self, inputs):
        x = self.conv2D(inputs)
        x = self.activ(x)

        if self.batch_norm:
            x = self.norm_layer(x)

        output = self.drop(x)

        return output
