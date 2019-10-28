import tensorflow as tf

class LeNet5(tf.keras.Model):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=6,kernel_size=(3,3),strides=1,padding="valid",
            input_shape=(32,32,1),activation=tf.nn.relu,name="conv1_layer")

        self.avg_layer_1 = tf.keras.layers.AvgPool2D(
            pool_size=2,strides=2,padding="valid")

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=16,kernel_size=(3,3),strides=1,padding="valid",
            activation=tf.nn.relu,name="conv2_layer")

        self.avg_layer_2 = tf.keras.layers.AvgPool2D(
            pool_size=2,strides=2,padding="valid")

        self.flatten_layer = tf.keras.layers.Flatten()

        self.fc_layer_1 = tf.keras.layers.Dense(
            units=120,activation=tf.nn.relu,name="fc1_layer")

        self.fc_layer_2 = tf.keras.layers.Dense(
            units=84,activation=tf.nn.relu,name="fc2_layer")

        self.fc_output_layer = tf.keras.layers.Dense(
            units=10,activation=tf.nn.softmax,name="fc3_output_layer")


    def call(self, x, training=False):
        x = self.conv_layer_1(x)
        x = self.avg_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.avg_layer_2(x)
        x = self.flatten_layer(x)
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        x = self.fc_output_layer(x)
        return x