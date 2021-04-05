from src.constants import constants as C

HTTP = "http://"
LOCALHOST = "localhost"
LOCALPORT = 8081
PORTRABBIT = 5672

REMOTEHOST = "" # TODO -> "http://yourremotehost.com"
REMOTEPORT = 0 # TODO -> your port number

adaptative_aggregation = True
min_grads_to_accumulate = 2
limit_outdated_gradients = 5

max_workers = 64
gradients_to_accumulate = 32 # This parameter is useless when we use adaptive aggregation.
local_steps = 50


DEFAULT_JOB = {
	"id_job": "",
	"id_initiator": 0,
	"description_job": "async_nn",
	"data": {
		"host_local": HTTP + LOCALHOST,
		"port_local": LOCALPORT,
		"host_remote": HTTP + REMOTEHOST,
		"port_remote": REMOTEPORT,
		"dataset": C.MNIST,
		"shape": [-1, 28, 28, 1],
		"mb_size": 8,
		"train_dataset_len": 60000,
		"test_dataset_len": 10000,
		"local_portion_dataset": max_workers,
		"num_classes": 10,
		"dtype": "float32"
	},
	"termination_criteria": {
		"global_max_age_model": 300
	},
	"topology": {
		"host_local": HTTP + LOCALHOST,
		"port_local": LOCALPORT,
		"host_remote": HTTP + REMOTEHOST,
		"port_remote": REMOTEPORT,
		"topology": C.MNIST_CONV_28_28_1
	},
	"weights": {
		"host_local": HTTP + LOCALHOST,
		"port_local": LOCALPORT,
		"host_remote": HTTP + REMOTEHOST,
		"port_remote": REMOTEPORT,
	},
	"gradients": {
		"host_local": HTTP + LOCALHOST,
		"port_local": LOCALPORT,
		"host_remote": HTTP + REMOTEHOST,
		"port_remote": REMOTEPORT,
	},
	"optimizer": {
		"class_name": "RMSprop",
		"config": {
			"name": "RMSprop",
			"learning_rate": 0.001,
			"decay": 0.0,
			"rho": 0.9,
			"momentum": 0.0,
			"epsilon": 1e-08,
			"centered": False
		}
	},
	"aggregator": {
		"gradients_to_accumulate": gradients_to_accumulate,
		"limit_outdated_gradients": limit_outdated_gradients,
		"adaptative_aggregation": adaptative_aggregation,
    "min_grads_to_accumulate": min_grads_to_accumulate,
    "max_grads_to_accumulate": max_workers,
		"rabbit_host": LOCALHOST,
		"rabbit_port": PORTRABBIT
	},
	"worker": {
		"local_computation": True,
 		"local_steps": local_steps
	},
	"tester": {
		"metric": {"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}},
 		"losses": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}
	},
  "servers": {

  }
}
