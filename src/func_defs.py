from all_imports import *

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Sparse
optimizer = tf.keras.optimizers.Adam()
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, questions, answers ,model):
  with tf.GradientTape() as tape:
    # Forward pass
    predictions = model(images, questions)
    train_loss = loss_function(y_true=answers, y_pred=predictions)

  # Backward pass
  gradients = tape.gradient(train_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  # Record results
  train_loss_metric(train_loss)
  train_accuracy_metric(answers, predictions)

def test_step(images,questions, answers,model):
  predictions = model(images,questions)
  test_loss = loss_function(y_true=answers, y_pred=predictions)

  # Record results
  test_loss_metric(test_loss)
  test_accuracy_metric(answers, predictions)
  return predictions

@tf.function
def train_step_state(images, questions, answers, hidden ,model):
  with tf.GradientTape() as tape:
    # Forward pass
    predictions = model(images, questions, hidden)
    train_loss = loss_function(y_true=answers, y_pred=predictions)

  # Backward pass
  gradients = tape.gradient(train_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  # Record results
  train_loss_metric(train_loss)
  train_accuracy_metric(answers, predictions)
