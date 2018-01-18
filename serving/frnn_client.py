#!/usr/bin/env python2.7
from __future__ import print_function

import sys, os
import numpy as np
#import threading
from functools import partial

from grpc.beta import implementations

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from plasma.preprocessor.preprocess import guarantee_preprocessed
from plasma.conf import conf
from plasma.models.loader import Loader
#from plasma.models.loader import ProcessGenerator
from plasma.preprocessor.normalize import Normalizer

if conf['data']['normalizer'] == 'minmax':
    from plasma.preprocessor.normalize import MinMaxNormalizer as Normalizer
elif conf['data']['normalizer'] == 'meanvar':
    from plasma.preprocessor.normalize import MeanVarNormalizer as Normalizer
elif conf['data']['normalizer'] == 'var':
    from plasma.preprocessor.normalize import VarNormalizer as Normalizer
elif conf['data']['normalizer'] == 'averagevar':
    from plasma.preprocessor.normalize import AveragingVarNormalizer as Normalizer
else:
    print('unkown normalizer. exiting')
    exit(1)

def main(_):
    normalizer = Normalizer(conf)
    normalizer.train()
    loader = Loader(conf,normalizer)
    shot_list_train,shot_list_validate,shot_list_test = guarantee_preprocessed(conf)
    batch_generator = partial(loader.training_batch_generator_partial_reset,shot_list=shot_list_train)
    #ProcessGenerator(batch_generator())
    batch_iterator_func = batch_generator()
    for i in range(conf['serving']['num_tests']):
        batch_xs,_,_,_,_,_ = next(batch_iterator_func)

        hostport = conf['serving']['server']
        host, port = hostport.split(':')
        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'frnn_model'
        request.model_spec.signature_name = 'predict'
        request.inputs['shots'].CopyFrom(tf.contrib.util.make_tensor_proto(batch_xs,dtype=tf.float32))

        prediction = stub.Predict(request, conf['serving']['request_freq'])
        result = np.expand_dims(prediction.outputs['scores'].float_val, axis=0).reshape(conf['training']['batch_size'],conf['model']['length'],1)
        print("Inference request {} {}".format(i,result))

if __name__ == '__main__':
    tf.app.run()
