#!/bin/bash
K=/Users/bhanin/Downloads/my-key.pem
DST=/Users/bhanin/Desktop/Experiments/ai-theorist/rounds/010-overnight
for i in $(seq 1 84); do
  ok=1
  for f in big_out.json big.log big_out_v1.json big_v1.log; do
    scp -q -i $K -o ConnectTimeout=20 -o StrictHostKeyChecking=no \
        ubuntu@34.210.24.111:/home/ubuntu/$f $DST/$f 2>/dev/null || ok=0
  done
  [ $ok -eq 1 ] && echo "$(date +%H:%M) pulled ok ($(wc -l < $DST/big.log 2>/dev/null) log lines)" >> $DST/poll.log \
                || echo "$(date +%H:%M) partial/failed pull" >> $DST/poll.log
  sleep 600
done
