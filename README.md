# CSE151A Project
## Introduction

Effective waste management is essential to environmental health. Improper waste management contributes to pollution, the spread of disease, contamination, ecosystem degradation, and worsened climate change. A key step in waste management starts by properly identifying waste as recyclable, landfill, or compost. However, due to the barrier of improper identification of waste items and non-immediate consequences, people often dispose of waste in the wrong containers. Our project applies machine learning-based image classification techniques to automate the identification and sorting of waste into landfill, recyclable, or compost categories. This will improve waste management by automating the sorting process, increasing accuracy, and preventing items like recyclables from being incorrectly sent to landfills. In the long-term, we hope that projects like ours enhance waste management systems, minimize environmental harm, and contribute to the development of a more sustainable future.

## Methods

### Data Exploration

The dataset has 15,000 pictures of waste classified as plastic, paper, cardboard, glass, metal, organic waste, and textiles intercepted already in a landfill environment. This will help the accuracy of our model since it is devoid of surrounding trash. The plots and results from data exploration can be seen below:

- Number of classes: 30
- Number of images: 15000
- Unique image sizes: [(256, 256)]
- Number of images per class: 500

The distribution of image sizes showing that each image was 256 by 256 pixels:

![Image](./images/imagesize_distribution.png)

The distribution to see the number of images per class:

![Image](./images/imagesperclass.png)

A sample image from each class is below:

![Image](./images/sampleimageseachclass.png)

### Preprocessing
We classified waste items (the existing 30 waste categories) into these three general categories:
- **Landfill** - Items that are non-recyclable or non-compostable and should be disposed of in landfills.
- **Recyclable** - Items that can be recycled, such as plastics, metals, glass, and paper products.
- **Compost** - Items that can decompose and be used as compost.

| label | category/class |
|:------|:---------------|
| `'aerosol_cans'` | if empty recyclable, otherwise landfill |
| `'aluminum_food_cans'` | recyclable |
| `'aluminum_soda_cans'` | recyclable |
| `'cardboard_boxes'` | recyclable |
| `'cardboard_packaging'` | recyclable |
| `'clothing'` | landfill |
| `'coffee_grounds'` | compost |
| `'disposable_plastic_cutlery'` | landfill |
| `'eggshells'` | compost |
| `'food_waste'` | compost |
| `'glass_beverage_bottles'` | recyclable |
| `'glass_cosmetic_containers'` | recyclable |
| `'glass_food_jars'` | recyclable |
| `'magazines'` | recyclable |
| `'newspaper'` | recyclable |
| `'office_paper'` | recyclable |
| `'paper_cups'` | recyclable as long as not wax coated |
| `'plastic_cup_lids'` | recyclable? |
| `'plastic_detergent_bottles'` | recyclable |
| `'plastic_food_containers'` | recyclable |
| `'plastic_shopping_bags'` | landfill |
| `'plastic_soda_bottles'` | recyclable |
| `'plastic_straws'` | recyclable |
| `'plastic_trash_bags'` | landfill |
| `'plastic_water_bottles'` | recyclable |
| `'shoes'` | landfill |
| `'steel_food_cans'` | recyclable |
| `'styrofoam_cups'` | landfill |
| `'styrofoam_food_containers'` | landfill |
| `'tea_bags'` | compost if plastic free |

We split our dataset into 60:20:20 for our training, validation, and test set.

We applied min-max normalization to the pixel data of the images, scaling each pixel value to be within the 0 to 1 range.

### Model 1

We used a **Random Forest** classifier for our first model to classify trash images. Our dataset contains a class imbalance, with recycling being the dominant class, which could potentially affect model performance. Given that Random Forest is suited for handling large and imbalanced datasets, we chose it as our initial model. The model was trained with basic parameters.

Our model had an accuracy of 0.8316 on our test set, meaning it had an error of 0.1684. After addressing overfitting in Milestone 4, it had an accuracy of 0.6677 on our test set.

This is how our model fit on the fitting graph:

![Fitting Graph](fitting_graph.png)
   
Our final model falls on the right of the fitting graph (created using our validation set) with the hyperparameter of n_estimators tuned to 80. 

### Model 2
Our second model was a convolutional neural network. The full code can be found [here](./CNN_supercomputer.ipynb).

We converted the split pandas datasets into a tensorflow dataset of rgb images: 
```
def to_tensorflow(df, shuffle=True):
    """
   Convert pandas df to tensorflow dataset.
    ARGS:
        df: pandas df
    RETURNS 
        gen: tensorflow dataset
    """
    preprocess = tf.keras.applications.vgg16.preprocess_input # preprocessing function for CNN 
    target_size=(224,224) # set the size of the images
    color_mode='rgb' # set the type of image
    class_mode= 'categorical' # set the class mode
    batch_size=32  # set the batch size 
    gen=ImageDataGenerator(preprocessing_function=preprocess).flow_from_dataframe(df, 
          x_col='image_path',
          y_col='category', target_size=target_size, color_mode=color_mode,
          class_mode=class_mode, batch_size=batch_size, shuffle=shuffle)
    return gen
pandas_dfs = [train_df, val_df]
train_batches, val_batches = [to_tensorflow(df) for df in pandas_dfs]
test_batches = to_tensorflow(test_df, shuffle=False) # don't shuffle test batch
```

We created a Sequential convolutional neural network model using keras. We used 2 convolutional hidden layers with a 3 X 3 kernel size and relu activation function, max pooling, and a Dense output layer with a softmax activation function:

![Model Summary](https://github.com/user-attachments/assets/027359d3-c358-4de7-9aa3-2dc40ca43ed2) <br/>*Convolutional Neural Network Model Summary*<br><br>

We compiled using cross entropy loss and a learning rate of 0.001. We trained the model with 20 epochs and a batch size of 1.

```
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=val_batches,
    validation_steps=len(val_batches),
    epochs=20,
    verbose=2,
    workers=-1,
    use_multiprocessing=True,
)
```

To better evaluate the the model fit, we plotted the training loss and validation loss at each epoch using the history of the fitted model. 
```
train_loss = history.history['loss']
val_loss = history.history['val_loss']
```
We used the fitted model to predict the test set.
```
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
```

To understand the predictions on the test set, we plotted the confusion matrix and printed the loss and accuracy across categories.

```
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
# CODE COPIED FROM SCI-KIT LEARN
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm_plot_labels = ['compost','landfill', 'recyclable']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
loss, accuracy = model.evaluate(test_batches, verbose=1)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy * 100:.2f}%')
```


### Model 3

## Results

## Discussion

## Conclusion

## Statement of Collaboration
