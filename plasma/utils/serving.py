#!/usr/bin/env python2.7
from __future__ import print_function

import sys
import threading

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from plasma.conf import conf

class ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self.num_tests = num_tests
    self.concurrency = concurrency
    self.error = 0
    self.done = 0
    self.active = 0
    self.condition = threading.Condition()

  def inc_error(self):
    with self.condition:
      self.error += 1

  def inc_done(self):
    with self.condition:
      self.done += 1
      self.condition.notify()

  def dec_active(self):
    with self.condition:
      self.active -= 1
      self.condition.notify()

  #modify 
  def get_error_rate(self):
    with self.condition:
      while self.done != self.num_tests:
        self.condition.wait()
      return self.error / float(self.num_tests)

  def throttle(self):
    with self.condition:
      while self.active == self.concurrency:
        self.condition.wait()
      self.active += 1


def create_rpc_callback(target, result_counter):
  """Creates RPC callback function.

  Args:
    target: The correct target for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['scores'].float_val)
      prediction = numpy.argmax(response)
      if target != prediction:
        result_counter.inc_error()
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def do_inference(test_gen):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test samples to use.
    test_gen: test batch generator

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  hostport = conf['serving']['server']
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = ResultCounter(conf['serving']['num_tests'], conf['serving']['concurrency'])

  for i in range(conf['serving']['num_tests']):
      batch_xs,batch_ys,_,_,_,_ = next(test_gen) #we need batch size to be 1 here!

      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'frnn_model'
      request.model_spec.signature_name = 'predict'
      request.inputs['shots'].CopyFrom(tf.contrib.util.make_tensor_proto(batch_xs,dtype=tf.float32))

      result_counter.throttle()
      result_future = stub.Predict.future(request, conf['serving']['request_freq'])
      result_future.add_done_callback(create_rpc_callback(batch_ys, result_counter))
      
  return result_counter.get_error_rate() 
