# cls_Melanoma
### Two-stage Deep Neural Network via Ensemble Learning for Melanoma Classification

<img src="https://user-images.githubusercontent.com/93422935/139534994-c7ef36a6-9491-4356-bccd-0c7b5e3c8ac4.png" alt="image-20211030213156848" width="80%" height="80%">

In this study, we propose an ensemble method that can integrate different types of classification networks for melanoma classification. Specifically, we first use U-net to segment the lesion area of images to generate a lesion mask, thus resize images to focus on the lesion; then, we use five excellent classification models to classify dermoscopy images, and adding squeeze-excitation block (SE block) to models to emphasize the informative features; finally we use our proposed new ensemble network to integrate five different classification results. The experimental results prove the validity of ourresults. We test our method on the ISIC 2017 challenge dataset, and obtain excellent results on multiple metrics, especially, we get 0.909 on ACC.

### Data

https://challenge.isic-archive.com/data/#2017

### Segmentation

```
# segment training data
python seg_train.py

# segment testing data
python seg_predict.py
```

### Resize 

```
resize.py
```

### Training

```
python training.py

# choose the network that you want to train here
# backbone_name = 'resnet50'
# backbone_name = 'densenet169'
backbone_name = 'inception_v3'
# backbone_name = 'inception_resnet_v2'
# backbone_name = 'xception'
```

### Predict

```
python cls_predict.py

# you should also specify the network name
```

### Vote

```
python vote.py

# training and testing

# network structure

inputs = keras.Input(shape=train_preds.shape[1:]) # 150x3x5
model = LocallyConnected1D(20, 1, activation='relu')(inputs) # 150x3x3
model = LocallyConnected1D(1, 1, activation='sigmoid')(model) # 150x3x1
model = keras.layers.Flatten()(model)
output = keras.layers.Softmax()(model)
model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer=Adam(lr=0.001), metrics=['accuracy'], loss='categorical_crossentropy')
model.summary()
model.fit(train_preds, label, epochs=20, validation_data=(val_preds, label_val))

model.save(os.path.join(model_data_dir, 'votetry.h5'))

```

### Results

####  Classification results with or without segmentation

<img src="https://user-images.githubusercontent.com/93422935/139535010-87ac16d6-75ab-4b1c-94e2-564d180e7a4e.png" alt="image-20211030213156848" width="50%" height="50%">

#### Two classification challenges

<img src="https://user-images.githubusercontent.com/93422935/139535069-e3db7a98-584a-4bbd-a6c0-8715644c5c37.png" alt="image-20211030213156848" width="50%" height="50%">

#### Compare with others

<img src="https://user-images.githubusercontent.com/93422935/139535098-84d6a4e2-03be-4ae9-86af-7be3d4538d83.png" alt="image-20211030213423575" width="50%" height="50%">

