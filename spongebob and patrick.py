import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from PIL import Image
import os
# An option for my particular PC. The code will run if you remove this.
tf.keras.backend.set_floatx('float64')
np.random.seed(326)
tf.random.set_seed(231)

# Setting up the training and testing data from hard drive ---------------------------------------
spongebob_image_data = list()
my_path = 'C:/Users/Jared/Documents/Python Scripts/spongebob/arranged data/Train/SpongeBob/'

img_dim = 50

i = -1
for file in os.listdir(my_path):
    i = i + 1
    spongebob_image_data.append(np.array(Image.open(my_path + file).convert('L')
                                         .resize((img_dim, img_dim), Image.ANTIALIAS)).reshape(img_dim, img_dim, 1))

spongebob_labels = [0 for i in range(len(spongebob_image_data))]

my_path = 'C:/Users/Jared/Documents/Python Scripts/spongebob/arranged data/Train/Patrick/'

patrick_image_data = list()

i = -1

for file in os.listdir(my_path):
    i = i + 1
    patrick_image_data.append(np.array(Image.open(my_path + file).convert('L')
                                       .resize((img_dim, img_dim), Image.ANTIALIAS)).reshape(img_dim, img_dim, 1))

patrick_labels = [1 for i in range(len(patrick_image_data))]

combined_training_data = np.stack(spongebob_image_data + patrick_image_data, axis=0)
combined_labels = np.array(spongebob_labels + patrick_labels)


x_train = combined_training_data * 1./255
y_train = combined_labels

perm = np.random.permutation(x_train.shape[0])
x_train = x_train[perm]
y_train = y_train[perm]

spongebob_image_data = list()
my_path = 'C:/Users/Jared/Documents/Python Scripts/spongebob/arranged data/Valid/SpongeBob/'

i = -1
for file in os.listdir(my_path):
    i = i + 1
    spongebob_image_data.append(np.array(Image.open(my_path + file).convert('L')
                                         .resize((img_dim, img_dim), Image.ANTIALIAS)).reshape(img_dim, img_dim, 1))

spongebob_labels = [0 for i in range(len(spongebob_image_data))]

my_path = 'C:/Users/Jared/Documents/Python Scripts/spongebob/arranged data/Valid/Patrick/'

patrick_image_data = list()

i = -1

for file in os.listdir(my_path):
    i = i + 1
    patrick_image_data.append(np.array(Image.open(my_path + file).convert('L')
                                       .resize((img_dim, img_dim), Image.ANTIALIAS)).reshape(img_dim, img_dim, 1))

patrick_labels = [1 for i in range(len(patrick_image_data))]

combined_test_data = np.stack(spongebob_image_data + patrick_image_data, axis=0)
combined_labels = np.array(spongebob_labels + patrick_labels)

x_test = combined_test_data * 1./255
y_test = combined_labels

perm = np.random.permutation(x_test.shape[0])
x_test = x_test[perm]
y_test = y_test[perm]
# -----------------------------------------------------------------------------------------------------------------


# Setting up our convolutional neural network --------------------------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(24, (3, 3), activation='relu', input_shape=(img_dim, img_dim, 1)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(24, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(2)
    ])

predictions = model(x_train[:1]).numpy()


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Training our network on our training data----------------------------------------------------------------

model.fit(x_train, y_train, batch_size=1, epochs=8)

print()
print('On the test set:')
model.evaluate(x_test, y_test, verbose=1)


n = np.random.randint(0, 10)
print('A random index for testing: ', n)
print(type(x_train[n,:,:]), x_test[n, :, :].shape)
print('Prediction:', tf.nn.softmax(model(x_test[n:n+1])).numpy())
print(y_test[n:n+1])
plt.imshow(Image.fromarray(x_test[n, :, :, 0].reshape(img_dim,img_dim)*255), cmap=matplotlib.cm.Greys_r)
plt.show()



