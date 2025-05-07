from tensorflow.keras.models import Sequential, load_model # to build sequential neural network 1:35:20
from tensorflow.keras.layers import LSTM, Dense, GRU, Bidirectional, TimeDistributed  # LSTM - temporal component for action detection, Dense - normal fully connected layer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint # used for logging to monitor model as it trains
# allow us to partition data into training and testing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# useful to convert all data into one hot encoded data
from tensorflow.keras.utils import to_categorical
from dataset import load_data
import os, time
import numpy as np, seaborn as sns, pandas as pd, matplotlib.pyplot as plt

def create_bi_model(input_dims, output_dims, activation='relu'):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation=activation), input_shape=input_dims))
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation=activation)))
    model.add(Bidirectional(LSTM(64, return_sequences=False, activation=activation)))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(output_dims, activation='softmax'))
    # adam optimiser [can change], categorical loss for multi-class clasification models, optional metrics 
    
    return model

def create_gru_model(input_dims, output_dims, activation='relu'):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, activation=activation, input_shape=input_dims))
    model.add(GRU(128, return_sequences=True, activation=activation))
    model.add(GRU(64, return_sequences=False, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(output_dims, activation='softmax'))
    
    return model
    
def create_model(input_dims, output_dims, activation='relu'):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation=activation, input_shape=input_dims))
    model.add(LSTM(128, return_sequences=True, activation=activation))
    model.add(LSTM(64, return_sequences=False, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(output_dims, activation='softmax'))
    # adam optimiser [can change], categorical loss for multi-class clasification models, optional metrics 
    
    return model
    

def create_and_train(actions, epochs=200, data_folder='mp_files', name='new_model', norm=True, hands_only=False, activation='relu', model_folder='_models', variables=[]):
    
    # Loading dataset
    sequences, labels = load_data(actions, data_folder, aug, norm, hands_only)
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    print(f"X:{len(X)}, y:{len(y)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=labels)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify=np.argmax(y_test, axis=1))
    my_vars = [X, y, X_train, y_train, X_test, y_test, X_val, y_val]
    np.save('_testing_data.npy', my_vars)
    
    
    model = create_model((X_train.shape[1], X_train.shape[2]), actions.shape[0], activation)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # outputs log of model training metrics
    log_dir = os.path.join('_models', model_folder, name, 'Logs')
    tb_callback = TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
    )
    
    # Saves model weights whenever there is an increase in validation accuracy
    checkpoint = ModelCheckpoint(
        filepath=os.path.join('_models', model_folder, name, f'{name}.h5'),
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[tb_callback, checkpoint]
    )
    
    # load the saved best weight
    model.load_weights(os.path.join('_models', model_folder, name, f'{name}.h5'))
    
    # eval_model(model, X_test, y_test)
  
    return model

def eval_model(model, X_test, y_test):
    # evaluate on all test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # confusion matrix
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # save confusion matrix as image
    cm_fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=actions, yticklabels=actions)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs(os.path.dirname('confusion_matrix.png'), exist_ok=True)
    plt.savefig('confusion_matrix.png')
    plt.close(cm_fig)

def my_load_model(load_path):
    if os.path.exists(load_path):
        model = load_model(load_path)
        return model
    else:
        print("Error - path not found!")
        return None


if __name__ == '__main__':
    actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M',
                        'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
    for aug in [True, False]:
        data_folder = '_models/_Aug_data/' if aug else '_models/_noAug_data/'
        mp_folder = 'standing_mp_sheyma' if aug else 'standing_mp_sheyma_noaug'
        sequences, labels = load_data(actions, mp_folder, aug, True, False)
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        print(f"X:{len(X)}, y:{len(y)}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=labels)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify=np.argmax(y_test, axis=1))
        my_vars = [X, y, X_train, y_train, X_test, y_test, X_val, y_val]
        
        
        # Save feature and label sets
        np.save(data_folder+'X_train.npy', X_train)
        np.save(data_folder+'X_test.npy', X_test)
        np.save(data_folder+'X_val.npy', X_val)
        np.save(data_folder+'y_train.npy', y_train)
        np.save(data_folder+'y_test.npy', y_test)
        np.save(data_folder+'y_val.npy', y_val)


        print("Datasets saved.")
