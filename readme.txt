# TIME_ATTACK README ---------------------------------------------------
To change the timing of the attack, you can go to line 258 and change the arguments of the conditional statement.The self.request_count parameter stands for the rounds number. Use this conditional statement to exclude the rounds you do not want to attack.E.g: If you want to attack round 1-20 out of 100 total, you should write self.request_count>20 to ensure the rounds greater than 20 are not attacked.

To run clients, do ./run_malicious.sh

Environment - This vision should be run on GPU instead of CPU !!! Greater graphical memory is anticipated.

# PROJECT README ---------------------------------------------------
fl_server.py needs no arguments, all the settings are in default_config.py
fl_client.py needs two parameters - client number and tier it belongs to. See run_clients.sh for details

fl_models.py now contains the models

Environment - m5d.24xlarge with 2 threads per core!!! This is important. Taskset has trouble with training keras models on single cpu for some reason. 

To run the server, do python3 fl_server.log
To run clients, do ./test_run_clients.sh

# OLD README ---------------------------------------------------
to run the clients, run ./run_clients.py [inital client number]
E.g. On node 1, we deploy clients 0-24 so command should be ./run_clients.py 0
On node 2, we deploy clients 25-49 so command should be ./run_clients.py 25

IF the clients on a node fail without any reason (happens often, probably becuase of cpulimit because this problem does not occur without it) STOP the server and clients and INCREASE the sleep time in the run_clients.sh files. That always helps.

IMP: client data are stored in logs/ folder to REMEMBER TO MOVE IT TO A DIFFERENT FOLDER before starting a new test.
Because the logs folder is cleared before every new run...


# OLD README ----------------------------------------
fl_server_old.py is for the old algorithm.
fl_server.py is for the new algorithm.
fl_client.py is used for both new and old algorithms.

You can set which dataset(MNIST or Cifar10) to use in the main function on both server side and client side.
The server side and client side dataset selection must be consistent. If you use MNIST on the server side, you should use MNIST on the client side too.

Also you should config the following two parameters in class FLServer in server side code.
IID = False # False means non-IID, True means IID
DATASET= 0 # 1 FOR MNIST, 0 FOR CIFAR10, DATASET must be consistent with the dataset setting in the main function.
