#! /bin/bash

sh /usr/local/spark/sbin/start-all.sh

sleep 1

ssh slave04 sh /home/packages/kafka_2.13-2.7.0/bin/zookeeper-server-start.sh /home/packages/kafka_2.13-2.7.0/config/zookeeper.properties &

sleep 1

ssh slave04 sh /home/packages/kafka_2.13-2.7.0/bin/kafka-server-start.sh /home/packages/kafka_2.13-2.7.0/config/server.properties &