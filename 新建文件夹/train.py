from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,\
 TensorBoard, CSVLogger
from keras.optimizers import Adam
from keras import backend as K
from models.model_big import first_model



trainset_dir = 'data/train/'
valset_dir = 'data/val/'
num_classes = 6
learning_rate = 1e-3
batch_size = 12
input_shape = (64, 64, 3)
momentum = 0.9



train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    trainset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size)

val_generator = val_datagen.flow_from_directory(
    valset_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=False)

K.clear_session()


optim = Adam(lr=learning_rate)
model = first_model(input_shape, num_classes)

model.compile(optimizer=optim, loss='categorical_crossentropy',
              metrics=['acc'])

csv_path = 'result_show/result_big/trash-classification6_gj-result.csv'
log_dir = 'result_show/log_big/'
save_weights_path = 'save_weights/model_big/trash-model-weight-ep-{epoch:02d}-val_loss-{val_loss:.4f}-val_acc-{val_acc:.4f}.h5'

checkpoint = ModelCheckpoint(save_weights_path, monitor='val_acc', verbose=1, 
                             save_weights_only=True, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=8, verbose=1)
earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=1)
logging = TensorBoard(log_dir=log_dir, batch_size=batch_size)
csvlogger = CSVLogger(csv_path, append=True)

callbacks = [checkpoint, reduce_lr, earlystop, logging, csvlogger]



num_epochs = 1000

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=num_epochs,
                    verbose=1, 
                    callbacks=callbacks, 
                    validation_data=val_generator, 
                    validation_steps=len(val_generator),
                    workers=1)
# fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, 
#               callbacks=None, validation_data=None, validation_steps=None, 
#               class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)