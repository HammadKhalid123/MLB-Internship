from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential

basemodel = MobileNetV2(weights='imagenet' , include_top=False)
basemodel.summary()
basemodel.trainable = False
model = Sequential([
    basemodel,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()