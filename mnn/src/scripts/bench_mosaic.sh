echo "" > tmp.txt
for MODEL in inception-v3 inception-v4 pnasnet-mobile pnasnet-large nasnet-mobile nasnet-large
do
  for THREAD in 1 2 4
  do
    ./mosaic_run_mnn.sh $MODEL $THREAD > tmp.txt
    sleep 10
    echo $MODEL-$THREAD
    cat tmp.txt | grep "avg"
  done
done
