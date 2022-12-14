# Report
Program 3 tasked me with the problem of building a neural network to predict the diagnosis regarding heart disease by specific critera based on clicical data; this involved cleaning the data, designing a nerual network, and validating the results.

## Data Preperation
Data preperation is just as important as the actual neural network. Without cleaned data, the NN is useless. The data I was given included 13 inputs of various types and a single output. 

The output variable was a value from 0-5, but I needed to identify the presence of a non-zero number.
```python
def update_labels(data: pd.DataFrame) -> pd.DataFrame:
    """ Updates the labels to be 0 or 1 """
    data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)
    return data
```

The next problem was the absence of some values in the data - stored as '?'. To minimize bias, I decided to simply remove the rows instead of replacing them with the mean or median, given some missing values came from rows that held classification data types instead of floats.

```python
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace('?', pd.NA)
    data = data.dropna()
    data = data.astype(float)
    return data
```

I then normalized the data; this allowed me to treat all inputs equally.

```python
def normalize(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in data.columns:
        if col in columns:
            # (x - min) / (max - min) -> [0, 1]
            data[col] = ((data[col] - data[col].min()) 
                / (data[col].max() - data[col].min()))
    return data
```

The next problem is to identify the most important inputs instead of trying to use every available input. Unfortuantley, I didn't make it this far in the assignment, for my priority was to get the neural network up and running first and foremost.

## Neural Network Configuration

The final neural network I wound up with was what you see below. I used a sigmoid activation function for the output layer, as I was trying to predict a binary value. I used a relu activation function for the hidden layers, as it is the most common activation function for hidden layers. I used a dropout layer to prevent overfitting.

The number of neurons per hidden layer was a bit of trial and error, and there was minimal research involved regarding the percentage of weights to drop. As for the number of layers, I let GitHub Copilot build the base design and I played around with it from there.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        8, activation='relu', input_shape=[len(train_data.keys())]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')])
```

## Training and Validation
Before doing any tranining, I set aside a portion of the data for validation. I used 20% of the given data for validation, and 80% for training / testing. I then split the training data into 80% for training and 20% for testing. 

```python
validation_data: pd.DataFrame = data.sample(frac=0.2, random_state=1)
data = data.drop(validation_data.index)

train_data: pd.DataFrame = data.sample(frac=0.8)
test_data: pd.DataFrame = data.drop(train_data.index)
```

For training and testing, I used the Adam optimizer and the binary crossentropy loss function. I used 400 epochs and a batch size of 20. The number of epochs seemed more than sufficient to train the data without overfitting. The batch size of 20 kept the training fast and minimized overfitting.

```python
history = model.fit(
    train_data, train_labels,
    epochs=400,
    batch_size=20,
)
```

## Results
Unfortuantley, the results were not up to my expectations and the variance of accuracy I expereinced between runs was underwhelming.

The model would range from mid-70% accuracy to high-80% accuracy run-to-run. While the high 80% accuracy isn't terrible, I was hoping the model would be a bit more consistent.

My final model with a test accuracy of 87.5% ran on the validation data had an accuracy of 88.14%.

## Comments
Overall, it was a neat problem. This isn't my first AI class or attempt at building a neural network, but I've never tried on a toy problem before, so it was nice to get a highesh accuracy with minimal effort. In the past I played around with sports predictions (baseball and esports) and man it is a lot harder haha :D

I'm not satisfied with the effort I gave to the project, though; I took a stab at it the week it was assigned, cleaned the data and ran the testing and what not, but I didn't touch it again until finals week; so, I never got to choosing specific inputs over others and messing around with learning rate and such.

## References
GitHub Copilot - AI writing AI models :)