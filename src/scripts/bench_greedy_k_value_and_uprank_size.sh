
MODEL=$1
MOBILE=$2
THREAD=$3
for SW in 4 6 8 10 12 14
do
  # python3 solver/greedy_device_placement.py $MODEL $MOBILE $THREAD --search_window=$SW > tmp.txt 2>&1
  # cat tmp.txt | grep "Greedy Result" | awk '{print $3}'
  python3 solver/ilp_device_placement.py $MODEL $MOBILE $THREAD --uprank_size=$SW  > tmp.txt 2>&1
  cat tmp.txt | grep "LP+serial+intersection" | awk '{print $6}'
done
