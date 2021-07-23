from kafka.admin import KafkaAdminClient
from kafka import KafkaProducer

import os
import ssl
import json
import time
import pandas as pd
from tqdm import tqdm

# SSL context to download without errors data from the given server
ssl._create_default_https_context = ssl._create_unverified_context

# define the kafka server from IP and Port
KAFKA_BOOTSTRAP_SERVERS = 'slave04:9092'

# producer definition from IP address given before
producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)


for i in range(0, 81):
    # Data download from s3 bucket
    i = str(i).zfill(2)
    url = f"https://cloud-areapd.pd.infn.it:5210/swift/v1/AUTH_d2e941ce4b324467b6b3d467a923a9bc/MAPD_miniDT_stream/data_0000{i}.txt"
    df = pd.read_csv(url)

    # Data cleaning for possible outliers
    df = df[df.ORBIT_CNT < 5e8]
    print(f"Reading file data_0000{i}.txt")

    # For loop over file size
    for j in tqdm(range(0, df.shape[0])):
        # dictionaries creation from dataframe's rows
        jj = df.iloc[j].to_dict()
        # unnecessary floats are cast into ints
        for key in ['HEAD', 'FPGA', "TDC_CHANNEL"]:
            jj[key] = int(jj[key])
        # json row is sent to the Kafka topic 'topic_stream'
        producer.send('topic_stream', json.dumps(jj).encode('utf-8'))
        time.sleep(0.0003)
    producer.flush()
