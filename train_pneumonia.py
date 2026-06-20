import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5

DATASET_PATH = os.path.join("dataset", "pneumonia")  # NORMAL / PNEUMONIA
SAVE_PATH = os.path.join("models", "pneumonia_model.h5")


def main():
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
    )

    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    os.makedirs("models", exist_ok=True)
    model.save(SAVE_PATH)
    print(f"\n✅ Saved: {SAVE_PATH}")
    print("Class indices:", train_data.class_indices)


if __name__ == "__main__":
    main()