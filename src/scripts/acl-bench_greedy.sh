
THREAD=2
for MODEL in inception_v3 inception_v4 pnasnet_large pnasnet_mobile nasnet_large
do
  adb shell "cd /data/local/tmp/ && acl-run.sh $MODEL CL parallel $THREAD greedy-placement-acl-$MODEL-$THREAD.txt"
done