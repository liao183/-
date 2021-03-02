import numpy as np 
import tensorflow as tf 
import seaborn as sn
from matplotlib import pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from models.model2_9 import first_model

input_shape = (256, 256, 3)
num_classes = 6
testset_dir = 'data/test/'
weight_path = 'save_weights/model2_9/trash-model-weight-ep-157-val_loss-0.3344-val_acc-0.9063.h5'
model =first_model(input_shape, num_classes)
model.load_weights(weight_path)

# Prediction on test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    testset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=233,
    class_mode='categorical')

x_test, y_test = test_generator.__getitem__(0)

test_true = np.argmax(y_test, axis=1)
test_pred = np.argmax(model.predict(x_test), axis=1)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_pred)))
print(classification_report(test_true, test_pred))

y_pred = model.predict(x_test)
acc = np.count_nonzero(np.equal(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)))/x_test.shape[0]
print(acc)# Test set accuracy

# Confusion Matrix and per-class accuracy
confusion_matrix = np.zeros((6,6), dtype=np.uint8)
per_class_acc = np.zeros(6)
for i in range(y_test.shape[1]):
    idxs = np.argmax(y_test, axis=1)==i
    this_label = y_test[idxs]
    num_samples_per_class = np.count_nonzero(idxs)
    one_hot = tf.one_hot(np.argmax(model.predict(x_test[idxs]), axis=1), depth=6).eval(session=tf.Session())
    confusion_matrix[i] = np.sum(one_hot, axis=0)
    per_class_acc[i] = confusion_matrix[i,i]/num_samples_per_class

LABELS=['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
plt.figure(figsize=(10,8))
sn.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt = 'd')
plt.title('Confusion Matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

accuracy = dict(zip(LABELS, per_class_acc))
print(accuracy)