import mlflow
import mlflow.tensorflow
import keras

#  Variables pour les param tres
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

# Lancement de la session de suivi MLflow
with mlflow.start_run():
 # Enregistrement des paramètres
 mlflow.log_param("epochs", EPOCHS)
 mlflow.log_param("batch_size", BATCH_SIZE)
 mlflow.log_param("dropout_rate", DROPOUT_RATE)

# Construction et entra nement du mod le (utiliser les variables finies)
# Chargement du jeu de donnees MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation des donnees
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Redimensionnement des images pour les reseaux fully-connected
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Construction du modèle
model = keras.sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrainement du modèle
history = model.fit(x_train, y_train, epochs=5, batch_size=120, validation_split=0.1)

# Evaluation du modèle
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Précision sur les données de test : {test_acc:.4f}")

# Sauvegarde du modèle
model.save("mnist_model.h5")
print("Modèle sauvegardée sous mnist_model.h5")

# Enregistrement des m triques
mlflow.log_metric("test_accuracy", test_acc)

# Enregistrement du mod le complet
mlflow.keras.log_model(model, "mnist-model")