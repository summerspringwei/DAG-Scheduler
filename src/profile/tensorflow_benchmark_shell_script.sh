============ MNIST LSTM Benchmark ================
adb shell /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_mnist_rnn_10000.pb --input_layer="images:0" \
--input_layer_shape="1,28,28" --input_layer_type="float" \
--output_layer="result_digit:0" --show_run_order=true --show_time=false \
--show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=8'

============ MobileNet Benchmark =================
adb shell 'taskset f0 /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_mobilenet_v1.pb --input_layer="input" \
--input_layer_shape="1,224,224,3" --input_layer_type="float" \
--output_layer="MobilenetV1/Predictions/Reshape_1" --show_run_order=true --show_time=false \
--show_memory=false --show_summary=true --max_time=10 --show_flops=true --num_threads=4'

adb shell 'taskset f0 /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_mobilenet_v2.pb --input_layer="input" \
--input_layer_shape="1,224,224,3" --input_layer_type="float" \
--output_layer="MobilenetV2/Predictions/Reshape_1" --show_run_order=true --show_time=false \
--show_memory=false --show_summary=true --max_time=10 --show_flops=true --num_threads=4'



============ GoogleNet Benchmark ==================
adb shell '/data/local/tmp/benchmark_model \
--graph=/mnt/sdcard/tensorflow/tensorflow_inception_graph.pb --input_layer="input" \
--input_layer_shape="1,224,224,3" --input_layer_type="float" \
--output_layer="output1" --show_run_order=true --show_time=false \
--show_memory=true --show_summary=true' --max_time=10 --show_flops=true --num_threads=8

adb shell 'taskset f0 /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_inception_v1.pb --input_layer="input" \
--input_layer_shape="1,224,224,3" --input_layer_type="float" \
--output_layer="InceptionV1/Logits/Predictions/Reshape_1" --show_run_order=true --show_time=false \
--show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4'

taskset f0 ./lite_benchmark_model --graph=/sdcard/tensorflow/inception-v1.tflite --input_layer='input' --input_layer_shape='1,224,224,3' --num_threads=4

adb shell 'taskset f0 /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_inception_v3.pb --input_layer="input" \
--input_layer_shape="1,299,299,3" --input_layer_type="float" \
--output_layer="InceptionV3/Predictions/Reshape_1" --show_run_order=true --show_time=false \
--show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4'

taskset f0 ./lite_benchmark_model --graph=/sdcard/tensorflow/inception-v3.tflite --input_layer='input' --input_layer_shape='1,299,299,3' --num_threads=4

============ ResNet_v1_50 Benchmark ==============
adb shell 'taskset f0 /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_resnet_v1_50.pb --input_layer="input" \
--input_layer_shape="1,224,224,3" --input_layer_type="float" \
--output_layer="resnet_v1_50/predictions/Reshape_1" --show_run_order=true --show_time=false \
--show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4'

adb shell 'taskset ff /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_resnet_v1_101_strip.pb --input_layer="input" \
--input_layer_shape="1,224,224,3" --input_layer_type="float" \
--output_layer="resnet_v1_101/predictions/Reshape_1" --show_run_order=true --show_time=false \
--show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=8'



taskset f0 ./lite_benchmark_model --graph=/sdcard/tensorflow/resnet-v1-50.tflite --input_layer='input' --input_layer_shape='1,224,224,3' --num_threads=8
========== VGG16 =======================
bazel run tensorflow/tools/benchmark:benchmark_model --graph=/home/xcw/datasets/tf_models/vgg16/frozen_vgg16.pb --show_flops --input_layer=input --input_layer_type=float --input_layer_shape=-1,224,224,3 --output_layer=vgg_16/fc8/BiasAdd

adb shell 'taskset f0 /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_vgg16.pb --input_layer="input" \
--input_layer_shape="1,224,224,3" --input_layer_type="float" \
--output_layer="vgg_16/fc8/BiasAdd" --show_run_order=true --show_time=false \
--show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4'

============= squeezenet-v11 ========================
bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/squeezenet_v11/frozen_squeezenet_v11.pb --show_flops --input_layer=image_placeholder,Placeholder --input_layer_type=float,float --input_layer_shape=1,227,227,3: --output_layer=Reshape_1 

adb shell 'taskset c0 /data/local/tmp/android_benchmark_model_1_7 --graph=/sdcard/tensorflow/frozen_squeezenet_v11.pb --show_flops --input_layer=image_placeholder,Placeholder --input_layer_type=float,float --input_layer_shape=1,227,227,3: --output_layer=Reshape_1 --num_threads=2'

bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=/home/xcw/datasets/tf_models/deeplab_v3_plus_mobilenet_v2/frozen_deeplab_v3_plus_mobilenet_v2.pb --show_flops --input_layer=sub_7 --input_layer_type=float --input_layer_shape=1,513,513,3 --output_layer=ResizeBilinear_2
taskset f0 ./lite_benchmark_model --graph=/sdcard/tensorflow/deeplab-v3-plus-mobilenet-v2.tflite --input_layer='sub_7' --input_layer_shape='1,513,513,3' --num_threads=4

============= frozen_deeplab_v3_plus_mobilenet_v2 ===============
adb shell 'taskset f0 /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_deeplab_v3_plus_mobilenet_v2.pb --input_layer="sub_7" \
--input_layer_shape="1,513,513,3" --input_layer_type="float" \
--output_layer="ResizeBilinear_2" --show_run_order=true --show_time=false \
--show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4'



==============pnasnet-mobile ===========
adb shell 'taskset f0 /data/local/tmp/android_benchmark_model_1_7 \
--graph=/mnt/sdcard/tensorflow/frozen_nasnet-mobile-mace-opt.pb --input_layer="input" \
--input_layer_shape="1,224,224,3" --input_layer_type="float" \
--output_layer="final_layer/predictions" --show_run_order=true --show_time=false \
--show_memory=false --show_summary=true --max_time=10 --show_flops=true --num_threads=4'




========== MLP_MNIST_1024_512_128 ================
adb shell "taskset f0 /data/local/tmp/benchmark_model --graph=/mnt/sdcard/tensorflow/frozen_mlp_mnist_1024_512_128.pb --input_layer="Placeholder" --input_layer_shape="1,784" --input_layer_type="float" --output_layer="logits/BiasAdd" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4"

adb shell "taskset 0c /data/local/tmp/android_benchmark_model_1_7 --graph=/mnt/sdcard/tensorflow/frozen_mlp_mnist_2048_4096_1024_inference.pb --input_layer="input" --input_layer_shape="1,784" --input_layer_type="float" --output_layer="logits/BiasAdd" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=2"

bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/nvidia/datasets/tf_models/mlp_mnist_big/frozen_mlp_mnist_2048_4096_1024_inference.pb --input_layer="input" --input_layer_shape="1,784" --input_layer_type="float" --output_layer="logits/BiasAdd" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true

=========== RNN_ptb_small Benchmark ===================
adb shell "taskset f0 /data/local/tmp/android_benchmark_model_1_7 --graph=/mnt/sdcard/tensorflow/frozen_rnn_ptn_small.pb --input_layer="Train/Model/input" --input_layer_shape="1,20,200" --input_layer_type="float" --output_layer="Train/Model/logits" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4"

========== SSD_MobileNet Benchmark =======================
adb shell "/data/local/tmp/benchmark_model --graph=/mnt/sdcard/tensorflow/ssd_mobilenet_v1_android_export.pb --input_layer="image_tensor" --input_layer_shape="1,300,300,3" --input_layer_type="uint8" --output_layer="detection_boxes,detection_scores" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true"


========== Alexnet benchmark ====================
adb shell "taskset f0 /data/local/tmp/android_benchmark_model_1_7 --graph=/mnt/sdcard/tensorflow/frozen_alexnet.pb --input_layer='InputData/X' --input_layer_shape="1,227,227,3"  --output_layer='FullyConnected_2/Softmax' --input_layer_type="float" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4"

========== SSD_MobileNet Benchmark ==============
adb shell "taskset ff /data/local/tmp/android_benchmark_model_1_7 --graph=/mnt/sdcard/tensorflow/frozen_ssd_mobilenet_v1.pb --input_layer="image_tensor" --input_layer_shape="1,300,300,3" --input_layer_type="float" --output_layer="detection_boxes,detection_scores" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=8"

========= shufflenet-v2-0.5 =====================
adb shell "taskset f0 /data/local/tmp/android_benchmark_model_1_7 --graph=/mnt/sdcard/tensorflow/frozen_shufflenet_v2.pb --input_layer='input' --input_layer_shape="1,224,224,3"  --output_layer='classifier/BiasAdd' --input_layer_type="float" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4"


========== Transformer ====================
/bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/Downloads/transfrom.pb --input_layer='input_tensor' --output_layer='model/Transformer/strided_slice_19' --input_layer_type="int64" --input_layer_shape="1,7"
========= Deep Speech =================
adb shell taskset ff /data/local/tmp/android_benchmark_model_1_7  --graph=/sdcard/tensorflow/frozen_deepspeech.pb --show_flops --input_layer=input_node,input_lengths,previous_state_c,previous_state_h --input_layer_type=float,int32,float,float --input_layer_shape=1,16,19,26:1:1,2048:1,2048 --output_layer=logits

./benchmark_model --graph=/sdcard/tensorflow/frozen_deepspeech.tflite --input_layer=input_node,previous_state_c,previous_state_h --input_layer_shape=1,16,19,26:1,2048:1,2048 --num_threads=4

=========== Server BenchMark =====================

============ ResNet_v1_50 Benchmark ==============
taskset -c 0 bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/resnet_v1_50/frozen_resnet_v1_50.pb --input_layer="input" --input_layer_shape="1,224,224,3" --input_layer_type="float" --output_layer="resnet_v1_50/predictions/Reshape_1" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=1

taskset -c 0 bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/resnet_v1_50/frozen_resnet_v1_50_inference_quantilized_opt.pb --input_layer="input" --input_layer_shape="1,224,224,3" --input_layer_type="float" --output_layer="resnet_v1_50/predictions/Reshape_1" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=1

============ GoogleNet Benchmark ================== 
bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/inception_v1/tensorflow_inception_graph.pb --input_layer="input" --input_layer_shape="1,224,224,3" --input_layer_type="float" --output_layer="output1" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true

taskset -c 0 bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/inception_v1/frozen_inception_v1.pb --input_layer="input" --input_layer_shape="1,224,224,3" --input_layer_type="float" --output_layer="InceptionV1/Logits/Predictions/Reshape_1" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=1

taskset -c 0 bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/inception_v3/frozen_inception_v3_without_squeeze.pb --input_layer="input" --input_layer_shape="1,299,299,3" --input_layer_type="float" --output_layer="InceptionV3/Predictions/Reshape_1" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=1


============ MobileNet Benchmark =================
taskset -c 0-15:2 bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/mobilenet_v1/frozen_mobilenet_v1_inference.pb --input_layer="input" --input_layer_shape="1,224,224,3" --input_layer_type="float" --output_layer="MobilenetV1/Predictions/Reshape_1" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=8

taskset -c 0-3 bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/mobilenet_v2/frozen_mobilenet_v2.pb --input_layer="input" --input_layer_shape="1,224,224,3" --input_layer_type="float" --output_layer="MobilenetV2/Predictions/Reshape_1" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4



============ MNIST LSTM Benchmark ================
bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/lstm/frozen_mnist_rnn_10000.pb --input_layer="images:0" --input_layer_shape="1,28,28" --input_layer_type="float" --output_layer="result_digit:0" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true


============ NSANet Benchmark ====================
bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/NASNet/nasnet_mobile.pb --input_layer="input" --input_layer_shape="1,224,224,3" --input_layer_type="float" --output_layer="final_layer/predictions" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true


=========== MLP MNIST Benchmark ==================
taskset -c 0-4 bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/mlp_mnist_model/frozen_mlp_mnist_1024_512_128.pb --input_layer="Placeholder" --input_layer_shape="1,784" --input_layer_type="float" --output_layer="logits/BiasAdd" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4

taskset -c 0-31:2 bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/mlp_mnist_big/frozen_mlp_mnist_2048_4096_1024.pb --input_layer="input" --input_layer_shape="1,784" --input_layer_type="float" --output_layer="logits/BiasAdd" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=16

========== RNN PTB Small Benchamrk ===============
taskset -c 0-4 bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/rnn_ptb/rnn_ptb_small/frozen_rnn_ptn_small.pb --input_layer="Train/Model/input" --input_layer_shape="1,20,200" --input_layer_type="float" --output_layer="Train/Model/logits" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=4

========== SSD_MobileNet Benchmark ===============
bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/ssd_mobilenet/ssd_mobilenet_v1_android_export.pb --input_layer="image_tensor" --input_layer_shape="1,300,300,3" --input_layer_type="uint8" --output_layer="detection_boxes,detection_scores" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true

========= Alexnet Benchmark ======================
bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/alexnet/frozen_alexnet.pb --input_layer='InputData/X' --input_layer_shape="1,227,227,3"  --output_layer='FullyConnected_2/Softmax' --input_layer_type="float" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true

========= facenet =================
bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=/home/xcw/datasets/tf_models/facenet/frozen_facenet-inception-resnet-v1.pb --input_layer='input,phase_train' --input_layer_shape="1,180,180,3:"  --output_layer='embeddings' --input_layer_type="float,bool" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true


=============== shufflenet v2 0.5 ===================
bazel-bin/tensorflow/tools/benchmark/benchmark_model  --graph=/home/xcw/datasets/tf_models/shufflenet_v2/frozen_shufflenet_v2.pb --input_layer='input' --input_layer_shape="1,224,224,3"  --output_layer='classifier/BiasAdd' --input_layer_type="float" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=1


============= Deep Speech =========================
tensorflow/tools/benchmark:benchmark_model -- --graph=/home/xcw/datasets/tf_models/deepspeech/models/output_graph.pb --show_flops --input_layer=input_node,input_lengths,previous_state_c,previous_state_h --input_layer_type=float,int32,float,float --input_layer_shape=1,16,19,26:1:1,2048:1,2048 --output_layer=logits


============= inception-resnet-v2 ================
bazel-bin/tensorflow/tools/benchmark/benchmark_model  --graph=/home/xcw/datasets/tf_models/inception-resnet-v2/frozen_inception-resnet-v2.pb --input_layer='input' --input_layer_shape="1,224,224,3"  --output_layer='InceptionResnetV2/Logits/Predictions' --input_layer_type="float" --show_run_order=true --show_time=false --show_memory=true --show_summary=true --max_time=10 --show_flops=true --num_threads=1


=========== Genrate Tensorflow Lite ============
bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=/home/xcw/datasets/tf_models/frozen_inception_v1_without_squeeze.pb \
  --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
  --output_file=/home/xcw/datasets/tf_models/google_inception/inception_v1.tflite --inference_type=FLOAT \
  --input_type=FLOAT --input_arrays=input \
  --output_arrays=InceptionV1/Logits/Predictions/Reshape_1 --input_shapes=1,224,224,3


bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=/home/xcw/datasets/tf_models/resnet/frozen_resnet_v1_50.pb \
  --input_format=TENSORFLOW_GRAPHDEF  --output_format=TFLITE \
  --output_file=/home/xcw/datasets/tf_models/resnet//resnet.tflite --inference_type=FLOAT \
  --input_type=FLOAT --input_arrays=input \
  --output_arrays=resnet_v1_50/predictions/Reshape_1 --input_shapes=1,224,224,3



============ Tensorflow Models export graph =========
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v1 \
  --output_file=/tmp/inception_v1_inf_graph_without_squeeze.pb

bazel-out/k8-py2-opt/bin/tensorflow/python/tools/freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph  --input_graph=/home/xcw/datasets/tf_models/vgg16/vgg16_inf_graph.pb   --input_checkpoint=/home/xcw/datasets/tf_models/vgg16/vgg_16.ckpt   --input_binary=true --output_graph=/home/xcw/datasets/tf_models/frozen_vgg16.pb  --output_node_names=vgg_16/fc8/squeezed

bazel-out/k8-py2-opt/bin/tensorflow/python/tools/freeze_graph --input_graph=/home/xcw/datasets/tf_models/resnet_v1_101/resnet_v1_101_inf.pb --input_checkpoint=/home/xcw/datasets/tf_models/resnet_v1_101/resnet_v1_101.ckpt --input_binary=true --output_graph=/home/xcw/datasets/tf_models/resnet_v1_101/frozen_resnet_v1_101.pb --output_node_names=resnet_v1_101/predictions/Reshape_1

bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=/home/xcw/datasets/tf_models/mlp_mnist_model/mlp_mnist_1024_512_128.pb   --input_checkpoint=/home/xcw/datasets/tf_models/mlp_mnist_model/mlp_mnist_model.ckpt   --input_binary=true --output_graph=/home/xcw/datasets/tf_models/mlp_mnist_model/frozen_mlp_mnist_1024_512_128.pb   --output_node_names=logits/BiasAdd

bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=/home/xcw/datasets/tf_models/rnn_ptb/rnn_ptb_small/graph.pbtxt   --input_checkpoint=/home/xcw/datasets/tf_models/rnn_ptb/rnn_ptb_small/model.ckpt-29646   --input_binary=false --output_graph=/home/xcw/datasets/tf_models/rnn_ptb/rnn_ptb_small/frozen_rnn_ptn_small.pb   --output_node_names=Train/Model/logits


bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=/home/xcw/datasets/tf_models/rnn_ptb/rnn_ptb_small/graph.pbtxt   --input_checkpoint=/home/xcw/datasets/tf_models/rnn_ptb/rnn_ptb_small/model.ckpt-33469   --input_binary=false --output_graph=/home/xcw/datasets/tf_models/rnn_ptb/rnn_ptb_small/frozen_rnn_ptn_small.pb   --output_node_names=Train/Model/logits


bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=/home/xcw/datasets/tf_models/mlp_mnist_model/mlp_mnist_1024_512_128.pb


==============

==================编译tensorflow mkl =================
bazel build --config=opt --config=mkl tensorflow/tools/benchmark:benchmark_model





bazel-bin/tensorflow/lite/toco/toco \
  --input_file=/home/xcw/datasets/tf_models/inception-resnet-v2/frozen_inception-resnet-v2.pb \
  --output_file=/home/xcw/datasets/tf_models/inception-resnet-v2/frozen_inception-resnet-v2-quantized.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape=1,224,224,3 \
  --input_array=input \
  --output_array=InceptionResnetV2/Logits/Predictions \
  --mean_value=128 \
  --std_value=127 \
  --default_ranges_min=0 \
  --default_ranges_max=255