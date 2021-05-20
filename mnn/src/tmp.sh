MOBILE=vivo_z3
for MODEL in acl-inception_v3 acl-inception_v4 acl-pnasnet_mobile \
  acl-pnasnet_large acl-nasnet_large acl-alexnet acl-mobilenet acl-mobilenet_v2
do
  if [ ! -d "../models/$MODEL/$MOBILE/" ]; then 
    mkdir ../models/$MODEL/$MOBILE/
  fi
  touch ../models/$MODEL/$MOBILE/$MODEL-$MOBILE-data-trans.csv
done
