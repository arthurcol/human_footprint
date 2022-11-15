import tensorflow as tf


class JaccardLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=0.5, smooth=1):
        super(JaccardLoss, self).__init__()
        # self.threshold = tf.constant([threshold])
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        # y_pred = tf.cast(tf.math.greater(y_pred,self.threshold),dtype=tf.int64) #TODO à éclaircir ?
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
        jaccard_unsmoothed = (intersection + self.smooth) / (
            sum_ - intersection + self.smooth
        )
        jaccad_smoothed = (1 - jaccard_unsmoothed) * self.smooth
        return tf.reduce_mean(jaccad_smoothed)


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=0.5):
        super(DiceLoss, self).__init__()
        # self.threshold = tf.constant([threshold])

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        # y_pred = tf.cast(tf.math.greater(y_pred,self.threshold),dtype=tf.int64) #TODO idem a éclaircir
        numerator = 2 * tf.reduce_sum(y_true * y_pred) + 1
        denominator = (
            tf.reduce_sum(y_true + y_pred) + 1
        )  # to avoid cases where denom == 0
        return 1 - numerator / denominator
