import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

imageHeight = 300
batchSize = 32
epochs = 50

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=25,
                                   horizontal_flip=True,
                                   # shear_range=0.2,
                                   # zoom_range=0.2,
                                   # width_shift_range=0.2,
                                   # height_shift_range=0.2,
                                   # vertical_flip=True,
                                   # fill_mode='nearest',
                                   )

validation_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(imageHeight, imageHeight),
                                                 batch_size=batchSize,
                                                 class_mode='categorical')

validation_set = validation_datagen.flow_from_directory('dataset/val',
                                                        target_size=(imageHeight, imageHeight),
                                                        batch_size=batchSize,
                                                        class_mode='categorical')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imageHeight, imageHeight, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 128 neuron hidden layers
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # 6 output neuron.
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_set,
                    epochs=epochs,
                    validation_data=validation_set)


