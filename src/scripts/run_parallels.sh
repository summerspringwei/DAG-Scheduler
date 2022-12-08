for thread in 1 2 4 ;
do
  for model in inception-v3 inception-v4 pnasnet-large pnasnet-mobile nasnet-large nasnet-mobile ;
  do
    echo "./run_mnn.sh $model $thread > tmp_$model_$thread.txt" ;
    ./run_mnn.sh $model $thread > tmp_$model_$thread.txt
    echo "tail -n 2 tmp_$model_$thread.txt" ;
    tail -n 2 tmp_$model_$thread.txt
    sleep 10
  done
done
    