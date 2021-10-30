import numpy as np

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  #指定要使用的GPU序号

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
        for gpu in gpu_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


if __name__ == '__main__':

    from keras import Model
    from models import backbone
    from paths import submission_dir
    from datasets.ISIC2018 import load_test_data, load_validation_data, load_training_data
    from misc_utils.prediction_utils import cyclic_stacking
    from misc_utils.eval_utils import get_precision_recall
    from sklearn.metrics import roc_curve, auc
    from scipy import interp

    def task3_tta_predict(model, img_arr):
        img_arr_tta = cyclic_stacking(img_arr)
        pred_logits = np.zeros(shape=(img_arr.shape[0], 7))

        for _img_crops in img_arr_tta:
            pred_logits += model.predict(_img_crops)

        pred_logits = pred_logits/len(img_arr_tta)

        return pred_logits

    backbone_name = 'inception_v3'
    # backbone_name = 'densenet169'
    # backbone_name = 'resnet50'
    # backbone_name = 'inception_resnet_v2'
    # backbone_name = 'xception'
    version = '1'
    use_tta = False
    num_classes = 3

    pred_set = 'test'  # or test
    load_func = load_validation_data if pred_set == 'validation' else load_test_data
    # load_func = load_training_data
    images, image_names, labels = load_func(task_idx=3, output_size=224)

    # images, image_names, labels =

    # max_num_images = 10
    max_num_images = images.shape[0]  # 1000
    images = images[:max_num_images]
    image_names = image_names[:max_num_images]

    # num_folds = 5

    # print('Starting prediction for set %s with TTA set to %r' % (pred_set, use_tta))


    y_pred = np.zeros(shape=(max_num_images, num_classes))
    score = 0

    # print('Processing fold ', k_fold)
    run_name = backbone_name + '_v' + version
    # run_name = 'task3_inception_v3_v3'
    model, _ = backbone(backbone_name).classification_model(load_from=run_name)

    predictions_model = Model(inputs=model.input, outputs=model.get_layer('predictions').output)
    from models import compile_model

    # compile_model(model=predictions_model, num_classes=num_classes, loss='crossentropy')
    from keras.optimizers import Adam
    from keras.losses import categorical_crossentropy
    predictions_model.compile(optimizer=Adam(lr=0.0001), metrics=['accuracy'], loss=categorical_crossentropy)
    if use_tta:
        y_pred += task3_tta_predict(model=predictions_model, img_arr=images)
    else:
        y_pred += predictions_model.predict(images)
        print(predictions_model.evaluate(images, labels))
        precision, recall, f1, sp = get_precision_recall(labels, y_pred)
        # print(precision, recall)

    # ROC AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], y_pred[:, i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # print('specificity = ', 1-fpr["macro"])
    # print('sensitivity = ', tpr["macro"])
    print('auc = ', roc_auc["macro"])

    y_prob = softmax(y_pred)
    # print('****', y_pred)
    # print('_____', y_prob)

    print('Done predicting -- creating submission')
"""
    # submission_file = submission_dir + '/task3_' + pred_set + '_submission.csv'
    submission_file = submission_dir + '/' + backbone_name + '_test.csv'
    f = open(submission_file, 'w')
    # f.write('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
    f.write('image,MM,SK,BN\n')

    # for i in range(3):
    #     print(i)
    # print(max_num_images)
    # print(image_names)
    for i_image, i_name in enumerate(image_names):
        i_line = i_name
        # print(i_line)
        for i_cls in range(3):
            prob = y_prob[i_image, i_cls]
            if prob < 0.001:
                prob = 0.
            i_line += ',' + str(prob)

        i_line += '\n'
        f.write(i_line)  # Give your csv text here.

    # print('aaaa')
    f.close()
"""