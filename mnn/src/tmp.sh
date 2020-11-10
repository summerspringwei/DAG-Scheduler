python analyze/compare_latency.py inception-v3 vivo_z3 2 > tmp.txt
sleep 10
python analyze/compare_latency.py inception-v4 vivo_z3 2 >> tmp.txt
sleep 10
python analyze/compare_latency.py pnasnet-mobile vivo_z3 2 >> tmp.txt
sleep 10
python analyze/compare_latency.py pnasnet-large vivo_z3 2 >> tmp.txt
sleep 10
python analyze/compare_latency.py nasnet-mobile vivo_z3 2 >> tmp.txt
sleep 10
python analyze/compare_latency.py nasnet-large vivo_z3 2 >> tmp.txt
sleep 10

