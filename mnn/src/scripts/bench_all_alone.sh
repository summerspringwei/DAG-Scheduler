
for MODEL in inception_v3 inception_v4 pnasnet_large pnasnet_mobile nasnet_large
do
  ./scripts/bench_alone.sh $MODEL vivo_z3
done