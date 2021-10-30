import keras
from models import Backbone
from keras.applications import nasnet


class NasnetBackbone(Backbone):
    def __init__(self, backbone_name='nasnet', **kwargs):
        super(NasnetBackbone, self).__init__(backbone_name)
        self.custom_objects['nasnet'] = nasnet

    def build_base_model(self, inputs, **kwarg):
        if self.backbone_name == 'nasnet':
            inputs = keras.layers.Lambda(lambda x: nasnet.preprocess_input(x))(inputs)
            Nasnet = nasnet.NASNetLarge(include_top=False,
                                        weights='imagenet',
                                        input_tensor=inputs)
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return Nasnet

    def classification_model(self,
                             num_dense_layers=0,
                             num_dense_units=0,
                             dropout_rate=0.2,
                             pooling='avg',
                             name='default_inception_resnet_classification_model',
                             **kwargs):
        """ Returns a classifier model using the correct backbone.
        """

        return super(NasnetBackbone, self).classification_model(num_dense_layers=num_dense_layers,
                                                                num_dense_units=num_dense_units,
                                                                dropout_rate=dropout_rate,
                                                                pooling=pooling,
                                                                name=name, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['nasnet', ]

        if self.backbone_name not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name,
                                                                                       allowed_backbones))
