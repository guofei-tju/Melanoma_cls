if __name__ == '__main__':

    from datasets.ISIC2018 import *
    from models import backbone
    from callback import config_cls_callbacks
    from misc_utils.eval_utils import compute_class_weights
    from misc_utils.print_utils import log_variable, Tee
    from misc_utils.filename_utils import get_log_filename
    from misc_utils.visualization_utils import BatchVisualization
    from keras.preprocessing.image import ImageDataGenerator

    import sys

    import os
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定要使用的GPU序号

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)

    # backbone_name = 'vgg16'
    # backbone_name = 'resnet50'
    # backbone_name = 'densenet169'
    backbone_name = 'inception_v3'
    # backbone_name = 'inception_resnet_v2'
    # backbone_name = 'xception'
    # backbone_name = 'nasnet'

    # Network architecture related params
    backbone_options = {}
    num_dense_layers = 1
    num_dense_units = 128
    pooling = 'avg'
    dropout_rate = 0.

    # Training related params
    dense_layer_regularizer = 'L1'
    class_wt_type = 'ones'
    # lr = 1e-4
    lr = 0.001
    slipt_at = 100

    version = '1'
    run_name = backbone_name + '_v' + version

    # Set prev_run_name to continue training from a previous run
    prev_run_name = None

    logfile = open(get_log_filename(run_name=run_name), 'w+')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)
    # sys.stdout = logfile

    # (x_train, y_train), (x_valid, y_valid), _ = load_training_data(task_idx=3,
    #                                                                output_size=224,
    #                                                                idx_partition=0)
    # x_train = load_task3_training_images(output_size=224)
    x_train = load_task3_resized_training_images(output_size=224)
    y_train = load_task3_training_labels()
    # x_valid = load_task3_validation_images(output_size=224)
    x_valid = load_task3_resized_validation_images(output_size=224)
    y_valid = load_task3_validation_labels()
    # print(y_valid)

    num_classes = y_train.shape[1]

    callbacks = config_cls_callbacks(run_name)

    model, base_model = backbone(backbone_name, **backbone_options).classification_model(
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
        num_dense_layers=num_dense_layers,
        num_dense_units=num_dense_units,
        pooling=pooling,
        dropout_rate=dropout_rate,
        kernel_regularizer=dense_layer_regularizer,
        save_to=run_name,
        load_from=prev_run_name,
        print_model_summary=True,
        plot_model_summary=False,
        lr=lr)

    n_samples_train = x_train.shape[0]
    n_samples_valid = x_valid.shape[0]

    class_weights = compute_class_weights(y_train, wt_type=class_wt_type)

    batch_size = 32
    use_data_aug = True
    horizontal_flip = True
    vertical_flip = True
    rotation_angle = 180
    width_shift_range = 0.1
    height_shift_range = 0.1

    log_variable(var_name='num_dense_layers', var_value=num_dense_layers)
    log_variable(var_name='num_dense_units', var_value=num_dense_units)
    log_variable(var_name='dropout_rate', var_value=dropout_rate)
    log_variable(var_name='pooling', var_value=pooling)
    log_variable(var_name='class_wt_type', var_value=class_wt_type)
    log_variable(var_name='dense_layer_regularizer', var_value=dense_layer_regularizer)
    log_variable(var_name='class_wt_type', var_value=class_wt_type)
    log_variable(var_name='learning_rate', var_value=lr)
    log_variable(var_name='batch_size', var_value=batch_size)

    log_variable(var_name='use_data_aug', var_value=use_data_aug)

    if use_data_aug:
        log_variable(var_name='horizontal_flip', var_value=horizontal_flip)
        log_variable(var_name='vertical_flip', var_value=vertical_flip)
        log_variable(var_name='width_shift_range', var_value=width_shift_range)
        log_variable(var_name='height_shift_range', var_value=height_shift_range)
        log_variable(var_name='rotation_angle', var_value=rotation_angle)

    log_variable(var_name='n_samples_train', var_value=n_samples_train)
    log_variable(var_name='n_samples_valid', var_value=n_samples_valid)

    sys.stdout.flush()  # need to make sure everything gets written to file
    sys.stdout = original

    if use_data_aug:

        datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip,
                                     width_shift_range=width_shift_range,
                                     height_shift_range=height_shift_range)
        generator_train = datagen.flow(x_train, y_train, batch_size=batch_size)
        # generator_valid = datagen.flow(x_valid, y_valid, batch_size=batch_size)

        # for layer in model.layers[:-4]:
        #     layer.trainable = False
        # for layer in model.layers[-4:]:
        #     layer.trainable = True
        # base_model.trainable = False

        # print(len(model.trainable_variables))
        # model.optimizer._set_hyper('learning_rate', 0.0001)
        from keras.optimizers import Adam

        model.compile(optimizer=Adam(lr=0.0001),
                      metrics=['accuracy'],
                      loss='categorical_crossentropy')
        # model.summary()

        model.fit_generator(generator=generator_train,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=100,
                            initial_epoch=0,
                            verbose=1,
                            validation_data=(x_valid, y_valid),
                            callbacks=callbacks)


        # for layer in model.layers[:-4]:
        #     layer.trainable = True

        # base_model.trainable = True
        # for layer in model.layers[:30]:
        #     layer.trainable = False
        # for layer in model.layers[30:]:
        #     layer.trainable = True

        # model.optimizer._set_hyper('learning_rate', 0.001)
        # model.compile(optimizer=Adam(lr=lr),
        #               metrics=['accuracy'],
        #               loss='categorical_crossentropy')
        # model.summary()
        #
        # model.fit_generator(generator=generator_train,
        #                     steps_per_epoch=x_train.shape[0] // batch_size,
        #                     epochs=100,
        #                     initial_epoch=0,
        #                     verbose=1,
        #                     validation_data=(x_valid, y_valid),
        #                     callbacks=callbacks)

    else:

        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=50,
                  verbose=1,
                  validation_data=(x_valid, y_valid),
                  class_weight=class_weights,
                  shuffle=True,
                  callbacks=callbacks)





