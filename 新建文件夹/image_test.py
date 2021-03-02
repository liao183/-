import keras.backend as K 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import pylab
from models.model2_6_9172 import first_model


K.clear_session()

input_shape = (64, 64, 3)
num_classes = 6
batch_size = 4

testset_dir = 'data/test/'

weight_path = 'save_weights/model2_6/trash-model-weight-ep-176-val_loss-0.33-val_acc-0.92.h5'
model = first_model(input_shape, num_classes)
model.load_weights(weight_path)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    testset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical')

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())


x_test, y_test = test_generator.__getitem__(0)
preds = model.predict(x_test)

plt.figure(figsize=(4, 4))
for i in range(batch_size):
    plt.subplot(2, 2, i+1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(y_test[i])]))
    # plt.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    plt.imshow(x_test[i])
pylab.show()





