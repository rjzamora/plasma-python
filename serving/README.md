#Real-time inference

TensorFlow serving Python api only works under Python2.7.
Following will export a pre-trained FRNN (Keras) model from a checkpoint:
```
python serving/frnn_saved_model.py
```

Now let's take a look at the export directory:
```
$ls /tigress/alexeys/export_frnn
1
```

Serving the inference in real-time requires a gRPC server. In the simples case, we start a local server at port 9000:

```
#bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=frnn_model --model_base_path=/tigress/alexeys/export_frnn
```

Finally, deploy:
```
python serving/frnn_client.py --num_tests=10 --server=localhost:9000
```
