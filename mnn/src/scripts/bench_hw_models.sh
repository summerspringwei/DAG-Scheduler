
DEVICE=$1
for MODEL in model1 model2 model3 model4
do
    for THREAD in 1 2
    do
        python solver/greedy_device_placement.py $MODEL $DEVICE $THREAD
        python visualization/draw_dag.py $MODEL $DEVICE $THREAD
    done
done

