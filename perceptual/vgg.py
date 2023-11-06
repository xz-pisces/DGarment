'''Image Transform Net and Loss Network models for Tensorflow.

Reference:
  - Justin Johnson, Alexandre Alahi and Li Fei-Fei. 
    [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](
      https://arxiv.org/abs/1603.08155) (ECCV 2016)

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Oct 2020
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16
# import tensorflow_addons as tfa



#
class LossNetwork(tf.keras.models.Model):
    def __init__(self, style_layers = ['block1_conv2',
                                       'block2_conv2',
                                       'block3_conv3',
                                       'block4_conv3']):
        super(LossNetwork, self).__init__()
        vgg = vgg16.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        model_outputs = [vgg.get_layer(name).output for name in style_layers]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)
        # mixed precision float32 output
        self.linear = layers.Activation('linear', dtype='float32')

    def forward(self, x):
        # x = vgg16.preprocess_input(x)
        x =  tf.image.resize(x, [512,512])
        x = self.model(x)
        return self.linear(x)

    def style_loss(self,style, output):
        return tf.add_n([tf.reduce_mean((style_feat - out_feat) ** 2)
                         for style_feat, out_feat in zip(style, output)])