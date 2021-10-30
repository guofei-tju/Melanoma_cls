import keras
import numpy as np
from paths import root_dir, model_data_dir
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling1D, LocallyConnected1D, Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets.ISIC2018 import load_task3_test_labels, load_task3_validation_labels, load_task3_training_labels
from misc_utils.eval_utils import get_precision_recall
from sklearn.metrics import roc_curve, auc
from scipy import interp
import os

if __name__ == '__main__':
    dir = root_dir + '/submissions'
    # label = load_task3_validation_labels()
    label = load_task3_training_labels()
    label_val = load_task3_validation_labels()
    label_test = load_task3_test_labels()
    backbones = ['inception_v3', 'resnet50', 'densenet169', 'xception', 'inception_resnet_v2']
    num = len(backbones)
    num_classes = 3

    print(np.array(label).shape)

    train_preds = []
    # labels = []
    for i, names in enumerate(backbones):
        pred = []
        with open(os.path.join(dir, (names + '.csv')), 'r') as f:
            for j, line in enumerate(f.readlines()[1:]):
                fields = line.strip().split(',')
                pred.append([eval(field) for field in fields[1:]])
                # pred.append(eval(fields[1]))
        train_preds.append(pred)
        # labels.append(label)

    val_preds = []
    for i, names in enumerate(backbones):
        pred = []
        with open(os.path.join(dir, (names + '_val.csv')), 'r') as f:
            for j, line in enumerate(f.readlines()[1:]):
                fields = line.strip().split(',')
                pred.append([eval(field) for field in fields[1:]])
                # pred.append(eval(fields[1]))
            val_preds.append(pred)

    test_preds = []
    for i, names in enumerate(backbones):
        pred = []
        with open(os.path.join(dir, (names + '_test.csv')), 'r') as f:
            for j, line in enumerate(f.readlines()[1:]):
                fields = line.strip().split(',')
                pred.append([eval(field) for field in fields[1:]])
                # pred.append(eval(fields[1]))
            test_preds.append(pred)

    train_preds = np.array(train_preds)
    print(train_preds.shape)
    train_preds = np.transpose(train_preds, [1, 2, 0])
    # train_preds = np.transpose(train_preds, [1, 0])

    val_preds = np.array(val_preds)
    print(val_preds.shape)
    val_preds = np.transpose(val_preds, [1, 2, 0])
    # val_preds = np.transpose(val_preds, [1, 0])

    test_preds = np.array(test_preds)
    print(test_preds.shape)
    test_preds = np.transpose(test_preds, [1, 2, 0])
    # test_preds = np.transpose(test_preds, [1, 0])
    # print(test_preds.shape)
    # labels = np.array(labels)
    # print(preds.shape)
    # print(label.shape)

    # model = Sequential()
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(num, activation='sigmoid'))


    # nerual network

    inputs = keras.Input(shape=train_preds.shape[1:]) # 150x3x5
    model = LocallyConnected1D(20, 1, activation='relu')(inputs) # 150x3x3
    model = LocallyConnected1D(1, 1, activation='sigmoid')(model) # 150x3x1
    model = keras.layers.Flatten()(model)
    output = keras.layers.Softmax()(model)
    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer=Adam(lr=0.001), metrics=['accuracy'], loss='categorical_crossentropy')
    model.summary()
    model.fit(train_preds, label, epochs=20, validation_data=(val_preds, label_val))
    #
    model.save(os.path.join(model_data_dir, 'votetry.h5'))

    # vote = model.get_layer('dense_1').get_weights()
    # print(len(vote))

    # svm
    # model = SVC()

    # model = RandomForestClassifier(n_estimators=50)

    # model = ExtraTreesClassifier(n_estimators=50)
    # model = KNeighborsClassifier(n_neighbors=8)
    # model = GradientBoostingClassifier()
    # model.fit(train_preds, label)

    y_pred = model.predict(test_preds)
    # print(y_pred)
    print(model.evaluate(test_preds, label_test))
    # print('acc = ', model.score(test_preds, label_test))
    # print('precision = ', precision_score(label_test, y_pred))
    # print('recall = ', recall_score(label_test, y_pred))
    # print('f1 = ', f1_score(label_test, y_pred))
    # fpr, tpr, _ = roc_curve(label_test, y_pred)
    # print('auc = ', auc(fpr, tpr))

    precision, recall, f1, sp = get_precision_recall(label_test, y_pred)

    # ROC AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_pred[:, i], pos_label=1)
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


