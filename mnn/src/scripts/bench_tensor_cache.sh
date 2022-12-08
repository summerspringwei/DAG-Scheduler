for MODEL in inception-v3 inception-v4 pnasnet-mobile pnasnet-large nasnet-large
do
	./run_mnn_tensor_cache.sh $MODEL 2 > tmp.txt
	cat tmp.txt | grep "min"
	cat tmp.txt | grep "Use cached" | wc -l
	echo "" > tmp.txt
done
	
