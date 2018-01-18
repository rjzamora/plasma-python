import os
import numpy as np

import keras.backend as K
from keras.models import Model

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
#from tensorflow.contrib.session_bundle import exporter

from plasma.conf import conf
from plasma.models import builder
from plasma.models.runner import optimizer_class

#Load model
specific_builder = builder.ModelBuilder(conf)
model = specific_builder.build_model(False)
model.compile(optimizer=optimizer_class(),loss=conf['data']['target'].loss)
model.load_weights("/tigress/alexeys/model_checkpoints/model.143446175634456912039484164450187267403._epoch_.0.h5")

config = model.get_config()
weights = model.get_weights()
new_model = Model.from_config(config)
new_model.set_weights(weights)

#Set to zero
K.set_learning_phase(0)
export_path = os.path.join(conf['paths']['base_path'],conf['serving']['save_model_path'],conf['serving']['version'])
builder = saved_model_builder.SavedModelBuilder(export_path)
signature = predict_signature_def(inputs={'shots': new_model.input},
                                  outputs={'scores': new_model.output})
with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
       tags=[tag_constants.SERVING],
       signature_def_map={'predict': signature}) #this will set the request.model_spec.signature_name = 'predict'
    builder.save()
print ('Done exporting the model for serving! The model is in {}'.format(export_path))
