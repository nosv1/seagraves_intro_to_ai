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
            # (x - min) / (max - min) -> [0, 1]
            data[col] = ((data[col] - data[col].min()) 
                / (data[col].max() - data[col].min()))
    return data

def main() -> None:

    # age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num
    data: pd.DataFrame = pd.read_csv('heart-data/processed_cleveland.csv')
    data = clean_data(data)
    data = update_labels(data)
    data = normalize(data, [
        "age", "trestbps", "chol", "thalach", "oldpeak"])
        
    # DON'T TOUCH VALIDATION DATA UNTIL THE VERY VERY END
    validation_data: pd.DataFrame = data.sample(frac=0.2, random_state=1)
    data = data.drop(validation_data.index)

    train_data: pd.DataFrame = data.sample(frac=0.8)
    test_data: pd.DataFrame = data.drop(train_data.index)
    
    train_labels: pd.Series = train_data.pop('num')
    test_labels: pd.Series = test_data.pop('num')
    validation_labels: pd.Series = validation_data.pop('num')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            8, activation='relu', input_shape=[len(train_data.keys())]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    # how to change learning rate
    # opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    # history = model.fit(
    #     train_data, train_labels,
    #     epochs=400,
    #     batch_size=20,
    # )
    

    # load model
    model = tf.keras.models.load_model('heart-model')

    test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=2)

    model.save('heart-model')
    print(f"Saving model with test accuracy: {test_accuracy}")

    # # predict the validation data
    # predictions = model.predict(validation_data)

    # correct = 0
    # for i in range(len(predictions)):
    #     if predictions[i] > 0.5:
    #         if validation_labels.iloc[i] == 1:
    #             correct += 1
    #     else:
    #         if validation_labels.iloc[i] == 0:
    #             correct += 1
    # print(f"Validation accuracy: {correct / len(predictions) * 100:.2f}%")

    # from ann_visualizer.visualize import ann_viz

    # # load model
    # model: tf.keras.Sequential = tf.keras.models.load_model('heart-model')

    # ann_viz(model, title="Heart Disease Model", view=False, filename="heart-model.gv")

    return None

if __name__ == "__main__":
    main()