set -xe
MODEL=$1
TARGET=$2
EXE=$3
THREADS=$4
MAP_FILE=$5

./graph_$MODEL --layout=NCHW --target=$TARGET --execution-type=$EXE --threads=$THREADS --device-map-file=$MAP_FILE --num_runs=10

