#! /bin/bash

sh /usr/local/spark/sbin/stop-all.sh

sleep 1

ssh slave04 sh /home/packages/kafka_2.13-2.7.0/bin/kafka-server-stop.sh &

sleep 5

ssh slave04 sh /home/packages/kafka_2.13-2.7.0/bin/zookeeper-server-stop.sh &
