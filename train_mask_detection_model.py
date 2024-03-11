from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# initial learning rate, epochs, and batch size
INIT_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

DIRECTORY = r"/Users/jatin/Documents/College/RT/face-mask-detection/combined_dataset"
CATEGORIES = ["with_mask", "without_mask"]


# initialize the list of class images
data = []
labels = []

for category in CATEGORIES:
	path = os.path.join(DIRECTORY, category)
	for img in os.listdir(path):
		img_path = os.path.join(path, img)
		image = load_img(img_path, target_size=(224, 224))
		image = img_to_array(image)
		image = preprocess_input(image)
		
		data.append(image)
		labels.append(category)

# applying encoding on labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# loading MobileNetV2 network
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

# actual model
model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
	layer.trainable = False

# compiling the model to train
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(
	aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	steps_per_epoch=len(trainX) // BATCH_SIZE,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BATCH_SIZE,
	epochs=EPOCHS)

# make predictions on testing data
predict = model.predict(testX, batch_size=BATCH_SIZE)
predict = np.argmax(predict, axis=1)

# classification report
print(classification_report(testY.argmax(axis=1), predict, target_names=lb.classes_))

# saving model to the disk
model.save("mask_detector.keras")

# plot the training loss and accuracy plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), history.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, EPOCHS), history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("AccuracyResults.png")

plt.figure()
plt.plot(np.arange(0, EPOCHS), history.history["loss"], label="Training Loss")
plt.plot(np.arange(0, EPOCHS), history.history["val_loss"], label="Validation Loss")
plt.title('Training and Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.savefig("LossResults.png")