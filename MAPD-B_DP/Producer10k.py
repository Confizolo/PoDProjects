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


for i in range(0, 81,4):
    # Data download from s3 bucket
    #i = str(i).zfill(2)
    url1 = f"https://cloud-areapd.pd.infn.it:5210/swift/v1/AUTH_d2e941ce4b324467b6b3d467a923a9bc/MAPD_miniDT_stream/data_0000{str(i).zfill(2)}.txt"
    url2 = f"https://cloud-areapd.pd.infn.it:5210/swift/v1/AUTH_d2e941ce4b324467b6b3d467a923a9bc/MAPD_miniDT_stream/data_0000{str(i+1).zfill(2)}.txt"
    url3 = f"https://cloud-areapd.pd.infn.it:5210/swift/v1/AUTH_d2e941ce4b324467b6b3d467a923a9bc/MAPD_miniDT_stream/data_0000{str(i+2).zfill(2)}.txt"
    url4 = f"https://cloud-areapd.pd.infn.it:5210/swift/v1/AUTH_d2e941ce4b324467b6b3d467a923a9bc/MAPD_miniDT_stream/data_0000{str(i+3).zfill(2)}.txt"

    df = pd.concat([pd.read_csv(url1), pd.read_csv(url2), pd.read_csv(url3), pd.read_csv(url4)])
    # Data cleaning for possible outliers
    df = df[df.ORBIT_CNT < 5e8]
    print(f"Converting file data_0000{i}.txt to data_0000{i+3}.txt")
    jj = df.to_dict('records')
    print(f"Reading file data_0000{i}.txt to data_0000{i+3}.txt")
    # For loop over file size
    for msg in tqdm(jj):
        # dictionaries creation from dataframe's rows
        # unnecessary floats are cast into ints
        # json row is sent to the Kafka topic 'topic_stream'
        producer.send('topic_stream', json.dumps(msg).encode('utf-8'))
#        time.sleep(0.00001)
    producer.flush()
