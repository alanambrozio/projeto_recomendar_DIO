import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model


def preprocess_images(data_dir, target_size=(224, 224)):
    images = []
    labels = []
    classes = os.listdir(data_dir)
    
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(label)
    
    return np.array(images), np.array(labels), classes

data_dir = r"D:\anaconda instalacoes\recomendar"
images, labels, class_names = preprocess_images(data_dir)
images = images / 255.0  # Normalizar os pixels para [0, 1]
print(f"Imagens carregadas: {images.shape}")
print(f"Classes: {class_names}")

# Carregar MobileNetV2 pré-treinada
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas da base
base_model.trainable = False

# Adicionar camadas personalizadas
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation="softmax")  # Uma saída para cada classe
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Treino: {x_train.shape}, Validação: {x_val.shape}")

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,  # Comece com 10 épocas
    batch_size=32
)

# Avaliação
val_loss, val_acc = model.evaluate(x_val, y_val)
print(f"Acurácia na validação: {val_acc:.2f}")

# Salvar o modelo
model.save(r"D:\anaconda instalacoes\recomendar\modelo_objetos.h5")

# Carregar modelo salvo
modelo = load_model("modelo_objetos.h5")

# Previsão
test_image = cv2.imread(r"D:\anaconda instalacoes\recomendar\teste.png")
test_image_resized = cv2.resize(test_image, (224, 224)) / 255.0
test_image_resized = np.expand_dims(test_image_resized, axis=0)

prediction = modelo.predict(test_image_resized)
predicted_class = class_names[np.argmax(prediction)]
print(f"Objeto reconhecido: {predicted_class}")