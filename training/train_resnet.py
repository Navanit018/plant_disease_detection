import tensorflow as tf

IMG_SIZE = (160,160)
BATCH = 32

train = tf.keras.utils.image_dataset_from_directory(
    "../dataset/train", image_size=IMG_SIZE, batch_size=BATCH)

val = tf.keras.utils.image_dataset_from_directory(
    "../dataset/val", image_size=IMG_SIZE, batch_size=BATCH)

NUM_CLASSES = len(train.class_names)

norm = tf.keras.layers.Rescaling(1./255)
train = train.map(lambda x,y:(norm(x),y))
val = val.map(lambda x,y:(norm(x),y))

base = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(160,160,3))

base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train, validation_data=val, epochs=10)
model.save("../models/resnet_model.keras")

print("✅ ResNet trained")
