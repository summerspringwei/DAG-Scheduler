
MOBILE=$1
for MODEL in inception-v3 inception-v4
do
    for THREAD in 1 2 4
    do
        echo "python generate_LP.py $MODEL $MOBILE $THREAD > result_data/$MOBILE-$MODEL-$THREAD-ilp_op_result.txt"
        python solver/generate_LP.py $MODEL $MOBILE $THREAD > result_data/$MOBILE-$MODEL-$THREAD-ilp_op_result.txt
    done
done