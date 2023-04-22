from itertools import count
from pickle import TRUE
from mediapipe_common import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import keras_tuner as kt

def model_builder(hp):
    inputs = tf.keras.Input(shape=(132))
    embedding = pose_embedder(inputs)
    
    layer = keras.layers.Dense(256, activation=tf.nn.relu6, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(embedding)
    layer = keras.layers.Dropout(0.5)(layer)

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    outputs = keras.layers.Dense(pose_loader.numberOfClasses, activation="softmax")(layer)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss= lossFunction,
        metrics=['accuracy'],
        #run_eagerly=True
    )

    return model



tuner = kt.Hyperband(model_builder,
                    objective='val_accuracy',
                    max_epochs=10,
                    factor=3,
                    directory='my_dir',
                    project_name='intro_to_kt')

tuner.search(pose_loader.x_input, pose_loader.y_labels, 
                epochs=50, 
                batch_size=8,
                class_weight = class_weights,
                validation_data = (pose_loader_test.x_input, pose_loader_test.y_labels), 
                callbacks=[earlystopping])