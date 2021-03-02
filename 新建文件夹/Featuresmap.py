from models.model2_6_9172 import first_model
from quiver_engine import server

input_shape=(64,64,3)
model = first_model(input_shape, 6)
model.load_weights('save_weights/model2_6/trash-model-weight-ep-176-val_loss-0.33-val_acc-0.92.h5')
server.launch(model, classes=['glass','cardboared','metai','paper','plasic','trash'], input_folder='hhh')
server.launch(model)

    