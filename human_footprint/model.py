import tensorflow as tf


class UnetLandcover(tf.keras.Model):
    def __init__(
        self,
        encoder="vgg16",
        pretrained=True,
        residual_connexion="concatenate",
        input_shape=(224, 224, 3),
    ):
        super().__init__()

        self.input_shape_ = input_shape

        if encoder == "vgg16":
            self.encoder = tf.keras.applications.vgg16.VGG16(
                include_top=False, weights="imagenet", input_shape=self.input_shape_
            )

        else:
            raise ValueError("WIP - only vgg16 for now")  ## ADD other backbones
        if pretrained == True:
            self.encoder.trainable = False

        if residual_connexion == "concatenate":
            self.residual_connexion = tf.keras.layers.Concatenate()

        elif residual_connexion == "add":
            self.residual_connexion = tf.keras.layers.Add()
        else:
            raise ValueError(
                "residual connexion must be one of concatenate or add"
            )  ## not implemented yet

        self.pretrained_encoder = self.create_pretrained_encoder(self.encoder)

        # layers
        # upsampling
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2))
        # conv2D
        self.conv1 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="sigmoid",
            name="conv1",
        )
        self.conv8 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="conv8",
        )
        self.conv32 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="conv32",
        )  # not used
        self.conv64 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="conv64",
        )
        self.conv128 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="conv128",
        )
        self.conv256 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="conv256",
        )
        self.conv512 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="conv512",
        )

    def call(self, inputs):
        output_encoder = self.pretrained_encoder(inputs)

        # central block
        x = self.conv512(output_encoder[-1])
        x = self.conv512(x)

        # decoder block5
        x = self.upsampling(x)
        x = self.conv512(x)

        # decoder block4
        x = self.upsampling(x)
        x = self.residual_connexion([x, output_encoder[-3]])
        x = self.conv256(x)

        # decoder block3
        x = self.upsampling(x)
        x = self.residual_connexion([x, output_encoder[-4]])
        x = self.conv128(x)
        # decoder block2
        x = self.upsampling(x)
        x = self.residual_connexion([x, output_encoder[-5]])
        x = self.conv64(x)

        # decoder block2
        x = self.upsampling(x)
        x = self.residual_connexion([x, output_encoder[-6]])
        x = self.conv8(x)

        output = self.conv1(x)
        return output

    @staticmethod
    def create_pretrained_encoder(model):
        skips = []
        prev_shape = 0
        for layer in model.layers:
            if layer.__class__.__name__ != "Conv2D":
                continue
            if layer.get_output_shape_at(0)[1:3] == prev_shape:
                continue
            skips.append(model.get_layer(layer.name).output)
            prev_shape = layer.get_output_shape_at(0)[1:3]

        last_layer_name = model.layers[-1].name

        return tf.keras.Model(
            inputs=model.inputs,
            outputs=skips + [model.get_layer(last_layer_name).output],
        )
