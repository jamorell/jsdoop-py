import signal
import sys

import json
import mysql.connector
from mysql.connector import errorcode

from src.api.load_job import load_job
from src.helper import url_solver as US
from src.constants import constants as C

print(sys.argv)
#0 file_name
#1 id_job


id_job = None
try:
  id_job = int(sys.argv[1]) #1234567890
except:
  exit("ERROR: Please insert a valid numeric job id.")


is_remote = False
try:
  is_remote = (sys.argv[2].lower() == 'true')
  print("REMOTE HOST")
except:
  print("LOCAL HOST")


# LOAD JOB
url_job = US.get_url_job(is_remote)
job_json = load_job(url_job, id_job)


myqueue = "stats_" + str(id_job)

mydb = None

try:
  mydb = mysql.connector.connect(
                              user=C.SQL_USER, password=C.SQL_PASSWORD,
                              host=C.SQL_HOST,
                              port=C.SQL_PORT,
                              database=C.SQL_DATABASE, auth_plugin='mysql_native_password')
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
  print("EXIT with error")
  sys.exit(0)
else:
  print("Connected to database")

total_to_sql = 10
list_values = []
to_ack = []
def callback(ch, method, properties, body):


  # Read the string representing json
  # Into a python list of dicts.
  myjson = json.loads(body)
  print("received " + str(myjson["timeRequest"]) + " " + str(myjson["typeTask"]))
  mycursor = mydb.cursor()
  sql = "INSERT IGNORE INTO stats(infoWorker, remoteAddr, timeRequest, timeResponse, ageModel, idJob, typeTask, other, username, idTask, executionTime) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
  values = (myjson["infoWorker"], myjson["remoteAddr"], myjson["timeRequest"], myjson["timeResponse"], myjson["ageModel"], myjson["idJob"], myjson["typeTask"], myjson["other"], myjson["username"], myjson["idTask"], myjson["executionTime"])
  list_values.append(values)
  to_ack.append(method.delivery_tag)
  if (len(list_values) >= total_to_sql):  
    #mycursor.execute(sql, values)
    mycursor.executemany(sql, list_values)
    mydb.commit()
    list_values.clear()
    for i in range(len(to_ack)):
      print("ACK " + str(to_ack[i]))
      channel.basic_ack(to_ack[i])
      #print("ACK " + str(method.delivery_tag))
      #channel.basic_ack(method.delivery_tag)
    to_ack.clear()

import pika

#connection = pika.BlockingConnection(pika.ConnectionParameters(host = "mallba9.lcc.uma.es", port = 4012))

connection = pika.BlockingConnection(pika.ConnectionParameters(*US.get_rabbit_params(job_json)))
channel = connection.channel()
channel.basic_qos(prefetch_count=(32)) # 1


#3- Subscribe to Results Queue
channel.queue_declare(queue=myqueue, durable=True)
channel.basic_consume(queue=myqueue,
                      auto_ack=False,
                      consumer_tag="a-consumer-tag",  
                      on_message_callback=callback)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()



def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    mydb.close()
    channel.close()
    connection.close()    
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
