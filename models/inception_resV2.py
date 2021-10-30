import keras
from keras.applications import inception_resnet_v2
from models import Backbone


class InceptionResBackbone(Backbone):

    def __init__(self, backbone_name='inception_resnet_v2', **kwargs):
        super(InceptionResBackbone, self).__init__(backbone_name)
        self.custom_objects['inception_resnet_v2'] = inception_resnet_v2

    def build_base_model(self, inputs, **kwargs):
        # create the inception backbone
        if self.backbone_name == 'inception_resnet_v2':
            inputs = keras.layers.Lambda(lambda x: inception_resnet_v2.preprocess_input(x))(inputs)
            inception_resnet = inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                                     input_tensor=inputs,
                                                                     weights='imagenet')
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return inception_resnet

    def classification_model(self,
                             num_dense_layers=0,
                             num_dense_units=0,
                             dropout_rate=0.2,
                             pooling='avg',
                             name='default_inception_resnet_classification_model',
                             **kwargs):
        """ Returns a classifier model using the correct backbone.
        """

        return super(InceptionResBackbone, self).classification_model(num_dense_layers=num_dense_layers,
                                                                      num_dense_units=num_dense_units,
                                                                      dropout_rate=dropout_rate,
                                                                      pooling=pooling,
                                                                      name=name, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['inception_resnet_v2', ]

        if self.backbone_name not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name,
                                                                                       allowed_backbones))