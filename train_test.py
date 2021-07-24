import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model

def parse_args():
    parser = argparse.ArgumentParser(description='Set a dataset')
    parser.add_argument('--half_channels', action='store_true', help='whether to make the num of feature channels half')
    parser.add_argument('--no_noise', action='store_true', help='whether not to include noise data')
    args = parser.parse_args()
    return args

args = parse_args()

# Set the dataset
dataset_dir = './dataset/'

if not args.no_noise:
    dataset_dir = dataset_dir + 'noise_'

x_train = np.load(dataset_dir + 'x_train.npy')
x_val = np.load(dataset_dir + 'x_val.npy')
y_train = np.load(dataset_dir + 'y_train.npy')
y_val = np.load(dataset_dir + 'y_val.npy')

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(y_train)).batch(32) # shuffleの引数をデータセット数と同じにすることで完全にshuffleされる.
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)

# Define the Network
class MyModel(Model):
  def __init__(self):
    super().__init__()
    if args.half_channels:        
        self.conv1 = Conv2D(64, 8, 1, padding='same')
        self.conv2 = Conv2D(128, 5, 1, padding='same')
        self.conv3 = Conv2D(64, 3, 1, padding='same')
    else:
        self.conv1 = Conv2D(128, 8, 1, padding='same')
        self.conv2 = Conv2D(256, 5, 1, padding='same')
        self.conv3 = Conv2D(128, 3, 1, padding='same')        
    self.batchnorm1 = BatchNormalization()
    self.batchnorm2 = BatchNormalization()
    self.batchnorm3 = BatchNormalization()
    self.activation = Activation('relu')
    self.pooling = GlobalAveragePooling2D()
    self.d1 = Dense(1, activation='sigmoid')

  def call(self, x):
    x = self.conv1(x)
    x = self.batchnorm1(x)
    x = self.activation(x)

    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = self.activation(x)

    x = self.conv3(x)
    x = self.batchnorm3(x)
    x = self.activation(x)    

    x = self.pooling(x)
    return self.d1(x)

model = MyModel()

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='validation_loss')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='validation_accuracy')

@tf.function
def train_step(time_series, labels):
  with tf.GradientTape() as tape:
    predictions = model(time_series)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def val_step(time_series, labels):
  predictions = model(time_series)
  v_loss = loss_object(labels, predictions)

  val_loss(v_loss)
  val_accuracy(labels, predictions)


# Train
EPOCHS = 200

plot_train_loss = []
plot_train_accuracy = []
plot_val_loss = []
plot_val_accuracy = []
for epoch in range(EPOCHS):
  for time_series, labels in train_ds:
    train_step(time_series, labels)

  for val_time_series, val_labels in val_ds:
    val_step(val_time_series, val_labels)

  if epoch >= 150:
    optimizer.lr = 0.0005
    if epoch >= 175:
      optimizer.lr = 0.0001

  plot_train_loss.append(float(train_loss.result()*100))
  plot_train_accuracy.append(float(train_accuracy.result()*100))
  plot_val_loss.append(float(val_loss.result()*100))
  plot_val_accuracy.append(float(val_accuracy.result()*100))
  template = 'Epoch {},  {}, Train Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         val_loss.result(),
                         val_accuracy.result()*100))

  train_loss.reset_states()
  train_accuracy.reset_states()
  val_loss.reset_states()
  val_accuracy.reset_states()

plot_x = [epoch + 1 for epoch in range(EPOCHS)]
plt.subplot(2,1,1)
plt.plot(plot_x, plot_train_accuracy, label="train accuracy")
plt.plot(plot_x, plot_val_accuracy, label="validation accuracy")
plt.ylabel('%')
plt.legend()
plt.subplot(2,1,2)
plt.plot(plot_x, plot_train_loss, label="train loss")
plt.plot(plot_x, plot_val_loss, label="validation loss")
plt.xlabel('Epoch')
plt.ylabel('%')
plt.legend()
plt.show()


#Test
def test_accuracy(time_series, labels):
  predictions = model(time_series)
  test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
  test_accuracy.reset_states()
  test_accuracy(labels, predictions)
  return test_accuracy.result()

logistic_x = np.load(dataset_dir + 'x_test.npy')
logistic_y = np.load(dataset_dir + 'y_test.npy')
sine_circle_x = np.load('./dataset/sine_circle_x_test.npy')
sine_circle_y = np.load('./dataset/sine_circle_y_test.npy')

print('Test the logistic Map')
print('Accuracy:', float(test_accuracy(logistic_x, logistic_y)), '%')
print('Test the Sine circle Map')
print('Accuracy:', float(test_accuracy(sine_circle_x, sine_circle_y)), '%')
