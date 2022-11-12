import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow as tf

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace('?', pd.NA)
    data = data.dropna()
    data = data.astype(float)
    return data

def update_labels(data: pd.DataFrame) -> pd.DataFrame:
    """ Updates the labels to be 0 or 1 """
    data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)
    return data

def normalize(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in data.columns:
        if col in columns:
            data[col] = ((data[col] - data[col].min()) 
                / (data[col].max() - data[col].min()))
    return data

def get_validation_data(
    data: pd.DataFrame, validation_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Returns validation data, remaining data """

    random.seed(1)
    validation_data: pd.DataFrame = data.sample(frac=validation_size)
    data = data.drop(validation_data.index)
    random.seed()

    return validation_data, data

def main() -> None:

    # age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num
    data: pd.DataFrame = pd.read_csv('heart-data/processed_cleveland.csv')
    data = clean_data(data)
    data = update_labels(data)
    data = normalize(data, [
        "age", "trestbps", "chol", "thalach", "oldpeak", "ca"])
        
    # DON'T TOUCH VALIDATION DATA UNTIL THE VERY VERY END
    validation_data, data = get_validation_data(data, 0.1)

    train_data: pd.DataFrame = data.sample(frac=0.8, random_state=1)
    test_data: pd.DataFrame = data.drop(train_data.index)
    
    train_labels: pd.DataFrame = train_data.pop('num')
    test_labels: pd.DataFrame = test_data.pop('num')
    validation_labels: pd.DataFrame = validation_data.pop('num')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            13, activation='relu', input_shape=[len(train_data.keys())]),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    model.fit(
        train_data, train_labels,
        epochs=400,
        batch_size=20,
    )

    test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=2)

    model.save('heart-model')
    
    return None

if __name__ == "__main__":
    main()