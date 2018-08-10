# Real-time inference with TensorFlow Serving

**Note**: TensorFlow serving Python api only works under Python2.7.
**Note**: this code is currently in a local `realtime_inference` branch, so you would need to:
```bash
git checkout realtime_inference
python setup.py install
```

This assumes you have already pre-trained a model following this tutorial:
https://github.com/PPPLDeepLearning/plasma-python/blob/master/docs/PrincetonUTutorial.md

## Configure serving

Following config parameter need to be specified in the `example/conf.yaml`:

```yaml
serving:
    concurrency: 1 #maximum number of concurrent inference requests
    num_tests: 10 #Number of test shots
    server: localhost:9000 #PredictionService host:port
    request_freq: 10.0 # 10 secs timeout
    version: '1' #if serving multiple versions of FRNN model
    p_threshold: -1.287 #best alarm threshold determined by performance analyzer
```

The important parameter here is the `p_threshold` which sets the best alarm threshold, determined from the ROC curve. Determined using the `performance_analyzer.py`.

Following will export a pre-trained FRNN (Keras) model from a checkpoint, and convert it into the protobuf format:
```
python serving/frnn_saved_model.py
```

The export directory `serving _model_checkpoints` will have a sub-folder for each versioni of pre-trained model served:
```
$ls /tigress/alexeys/serving _model_checkpoints
1
```

## Starting the (local) gRPC server 

Serving FRNN models in real-time requires a gRPC server. In this case, we spin up a gRPC local server at port `9000`:
```
#See: https://www.tensorflow.org/serving/serving_basic under "Load Exported Model With Standard TensorFlow ModelServer"
#bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=frnn_model --model_base_path=/tigress/alexeys/export_frnn
```

here the `--model_name=frnn_model` should match what you specify in the client code, which makes requests to the server (currently hardcoded, as we alsways serve the same model - FRNN):
```
request = predict_pb2.PredictRequest()
request.model_spec.name = 'frnn_model'
```

Finally, deploy client code:
```
python serving/frnn_client.py
```

The test-version of the code is only used as a trigger: ir only tells us whether disruption is coming and a type of prediction (true positive, false positive, etc). The code being developed in plasma/utils/serving.py module will also allow to count FP, TP and produce test level ROCs.
