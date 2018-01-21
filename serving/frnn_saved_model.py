import sys,os
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

import pdb

use_custom_path = len(sys.argv) > 1
custom_path = None
if use_custom_path:
    custom_path = sys.argv[1]
    print("Predicting using path {}".format(custom_path))

#Set to zero
K.set_learning_phase(0)

#Load model
specific_builder = builder.ModelBuilder(conf)
model = specific_builder.build_model(False,1)
model.compile(optimizer=optimizer_class(),loss=conf['data']['target'].loss)
#load the latest epoch we did. Returns -1 if none exist yet
e = specific_builder.load_model_weights(model,custom_path)
if e < 0:
    print("Pretrained model is not available. Train the model first!")
    sys.exit(1)

model.reset_states()

export_path = os.path.join(conf['paths']['serving_save_path'],conf['serving']['version'])
builder = saved_model_builder.SavedModelBuilder(export_path)
signature = predict_signature_def(inputs={'shots': model.input},
                                  outputs={'scores': model.output})
with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
       tags=[tag_constants.SERVING],
       #this will require set the request.model_spec.signature_name = 'predict'
       signature_def_map={'predict': signature})
    builder.save()

print ('Done exporting the model for serving! The model is in {}'.format(export_path))
