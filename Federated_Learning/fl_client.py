#!/usr/bin/python3
import numpy as np
import time
import keras
import random
from custom_logger import CustomLogger
import pickle
from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace
from fl_server import obj_to_pickle_string, pickle_string_to_obj

import logging
logging.getLogger('socketIO-client').setLevel(logging.ERROR)
logging.basicConfig()

import datasource
import sys
import os
import tensorflow as tf

class LocalModel(object):

    def __init__(self, model_config, trainx, trainy, logger):
        # model_config:
            # 'model': self.global_model.model.to_json(),
            # 'model_id'
            # 'min_train_size'
            # 'data_split': (0.6, 0.3, 0.1), # train, test, valid
            # 'epoch_per_round'
            # 'batch_size'
        self.model_config = model_config
        self.logger = logger

        # the weights will be initialized on first pull from server
        self.logger.debug("Initalizing model from server's json reply")
        #self.logger.debug("JSON :{}".format(model_config["model_json"]))
        self.model = model_from_json(model_config['model_json'])

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        self.logger.debug("Using RMSProp optimizer with {} lr".format(0.0001))

        # Let's train the model using RMSprop
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        self.x_train = trainx
        self.y_train = trainy
        self.logger.debug("Model initialized")

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self):
        self.model.fit(self.x_train, self.y_train,
                  epochs=self.model_config['epoch_per_round'],
                  batch_size=self.model_config['batch_size'],
                  verbose=0)
        return self.model.get_weights()

    def evaluate_on_data(self, x, y):
        self.logger.debug("Running evaluation on dataset...")
        score = self.model.evaluate(x, y, verbose=0)
        self.logger.debug("Scores (accuracy/loss): {:.3f}/{:.4f}".format(score[1], score[0]))
        return score


# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.
class FederatedClient(object):

    def __init__(self, config, client_num, tier):
        """
        Initilizer all object variables required for training. Also start communication with
        Server.
        """
        # LOGGER ---------------------------------------------------------------------------------
        # Adding parent folders to log files as part of full path
        if config["clients"]["log_file"]:
            config["clients"]["log_file"] = config["clients"]["log_file"].format(client_num, tier)
        if config["dataset"]["log_file"]:
            config["dataset"]["log_file"] = config["dataset"]["log_file"].format(client_num, tier)

        self.logger = CustomLogger(config["clients"]["logging_level"],
                                   "Client {}".format(client_num), config["clients"]["log_file"])
        self.config = config
        # ---------------------------------------------------------------------------------------
        self.judge = True
        self.client_offset = 0
        # CLIENT SPECS -----------------------------------------------------------------------------
        self.local_model = None
        self.client_num = client_num
        self.name = "client_{}".format(client_num)

        self.datasource = None # We should get the datasource from the Server

        self.tier = tier
        self.logger.info("Starting Client ->")
        self.logger.info("\tName: {}, Tier: {}".format(self.name, self.tier))
        # -------------------------------------------------------------------------------------------
        self.request_count = 0
        # SERVER CONNECTIONS -----------------------------------------
        server_host = config["clients"]["server_host"]
        server_port = config["clients"]["server_port"]
        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.logger.info("Starting Socket connection to Server"
                         "at {}:{}".format(server_host, server_port))
        #--------------------------------------------------------------
        
        # INITIALIZE SOCKET IO -------
        self.register_handles()
        self.logger.debug("Registered Socket Handles, sending \'wake up\' to server...")
        self.sio.emit('client_wake_up', {"name": self.name})
        self.sio.wait()
        # ----------------------------
        
        
    def reinit(self):
        
        self.judge = True
        
        self.local_model = None
        
        self.datasource = None # We should get the datasource from the Server
        
        self.logger.info("Reinit!!!!!!!!!!!!!!!!!!!!!")
        
        self.client_offset = self.client_offset+1
        
        self.register_handles()
        self.sio.emit('client_wake_up', {"name": self.name})
        self.sio.wait()
        
    
    ########## Socket Event Handlers ##########
    def on_init(self, *args):
        """
        Set up the model, training/test/validation data from recvieved and
        let Server know this Client is ready.
        """
        self.logger.debug("Server called to \'on_init\'")
        model_config = args[0]
        
        dataset = datasource.available_datasets[model_config['dataset']]
        self.datasource = dataset(self.config["dataset"])
        
        # Training data
        
        self.local_trainx = self.datasource.x_train[model_config['indices']]
        self.local_trainy = self.datasource.y_train[model_config['indices']]
        self.local_trainx, self.local_trainy = self.datasource.generate_data(
            self.local_trainx, self.local_trainy)

        # TODO: Add Poisoning
        # if sys.argv[3] == 'True':
        #     count = 0
        #     for ind in range(len(self.local_trainy)):
        #         if self.local_trainy[ind][0] == 1:
        #             count += 1
        #             self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]
                    
        #         if self.local_trainy[ind][2] == 1:
        #             count += 1
        #             self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]
        
        #         if self.local_trainy[ind][3] == 1:
        #             count += 1
        #             self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]

        #         if self.local_trainy[ind][4] == 1:
        #             count += 1
        #             self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]

        #         if self.local_trainy[ind][5] == 1:
        #             count += 1
        #             self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]
        #     self.logger.info("Count of poisoned imgs: {}".format(count))

        # Test Data
        self.test_x = self.datasource.x_train[model_config['test_indices']]
        self.test_y = self.datasource.y_train[model_config['test_indices']]
        self.test_x, self.test_y = self.datasource.generate_data(self.test_x, 
                self.test_y)
        
        # Validation data
        self.valid_x, self.valid_y = self.datasource.generate_data(
            self.datasource.x_test, self.datasource.y_test)

        self.logger.debug("Using dataset sizes: {} Training, "
                          "{} Test and {} Validation.".format(
                              len(self.local_trainx[0]), len(self.test_x[0]),
                              len(self.valid_x)))
        
        # Model
        self.local_model = LocalModel(model_config, self.local_trainx, 
                self.local_trainy, self.logger)
        self.logger.debug("Completed setting up model and training data")

        # Ready for training
        self.logger.info("Successfully connnected to Server and initialized model & dataset")
        self.logger.debug("Letting server know that \'client_ready\'...")
        self.sio.emit('client_ready', {
            'client_tier': self.tier, 'client_name': self.name, 
            'train_size': self.local_model.x_train.shape[0]})
        self.logger.debug("...Done!")
        self.logger.info("Waiting for instructions from Server...")


    def register_handles(self):
        """Register SocketIO Handles"""
        
        def on_connect():
            """Log that Connection to Serverwas made."""
            self.logger.info("Made a connection..")
       
        def on_disconnect():
            """On Server disconnection, terminate self"""
            self.logger.info("Disconnected from Server. Shutting down...")
            # exit(0)

        def on_connection_error(*args):
            """Log Connection Errors"""
            self.logger.info("Connection error")
            self.logger.debug("Connection Error msg:")
            self.logger.debug(str(args))
    
        def on_request_update(*args):
            """
            Update model with weights from server, train one round and run evaluations on
            requested datasets. Reply back with trained weights and acc/loss/time records.
            """
            if self.judge == False:
                self.reinit()
                
            else:
                self.judge = False
                self.request_count = self.request_count + 1
                self.logger.info("This is request_count")
                self.logger.info(self.request_count)
                req = args[0]
                self.logger.info("Update Requested by Server "
                                 "for Round # {}".format(req['round_number']))
                self.logger.info("Updating local weights and training...")
    
                # Update model with weights from server and train one round
                # while logging time.
                w_update_start_time = time.time()
                if req['weights_format'] == 'pickle':
                    weights = pickle_string_to_obj(req['current_weights'])
                self.local_model.set_weights(weights)
    
                # Train one round
                if self.request_count >5:
                    s_time = time.time()
                    my_weights = self.local_model.train_one_round()
                    e_time = time.time()
    
                else: 
                    if sys.argv[3] == 'True':
                        count = 0
                        for ind in range(len(self.local_trainy)):
                            if self.local_trainy[ind][0] == 1:
                                count += 1
                                self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]
                            
                            if self.local_trainy[ind][2] == 1:
                                count += 1
                                self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]
                
                            if self.local_trainy[ind][3] == 1:
                                count += 1
                                self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]
        
                            if self.local_trainy[ind][4] == 1:
                                count += 1
                                self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]
        
                            if self.local_trainy[ind][5] == 1:
                                count += 1
                                self.local_trainy[ind] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,]
                        self.logger.info("a local poisoning model: count of poisoned imgs: {}".format(count)) 
    
                        self.local_model.y_train=self.local_trainy
    
                        #self.local_model = LocalModel(self.modelconfig, self.local_trainx, self.local_trainy, self.logger)
    
                        self.logger.debug("Completed setting up a local poisoning model and training data")
                        self.logger.info("This is a poisoning client!!!!!!!!!!!!!!!")
                        s_time = time.time()
                        my_weights = self.local_model.train_one_round()
                        e_time = time.time()
                    else:
                        s_time = time.time()
                        my_weights = self.local_model.train_one_round()
                        e_time = time.time()
    
    
                self.logger.info("Finished training. Took {:.3f} s".format(e_time - s_time))
    
                # Run Evaluations ----------------------------------------------------------
                train_accuracy, test_accuracy, valid_accuracy = -1.0, -1.0, -1.0
                train_loss, test_loss, valid_loss = -1.0, -1.0, -1.0
                eval_start_time = time.time()
                train_loss, train_accuracy = self.local_model.evaluate_on_data(
                    self.local_trainx, self.local_trainy)
                end_time = time.time()
                self.logger.info("Training Evaluation ({:.2f}s)"
                                 " : {:.2f} %, {:.3f}".format((end_time - eval_start_time),
                                                              train_accuracy, train_loss))
                
                if req['run_validation']:
                    valid_loss, valid_accuracy = self.local_model.evaluate_on_data(
                        self.valid_x, self.valid_y)
                    end_time = time.time() - end_time
                    self.logger.info("Validation Evaluation ({:.2f}s)"
                                     " : {:.2f} %, {:.3f}".format(end_time, valid_accuracy, valid_loss))
                    
                if req['run_test']:
                    test_loss, test_accuracy = self.local_model.evaluate_on_data(
                        self.test_x, self.test_y)
                    end_time = time.time() - end_time
                    self.logger.info("Test Evaluation ({:.2f}s)"
                                     " : {:.2f} %, {:.3f}".format(end_time, test_accuracy, test_loss))
                # ---------------------------------------------------------------------------
                    
                self.logger.log_as_TSV(req['round_number'], "{:.3f}".format(end_time - w_update_start_time),
                        "{:2f}".format(train_accuracy), "{:.2f}".format(train_loss),
                        filename="./logs/runningtime_{}.txt".format(self.client_num))
    
                resp = {
                    'round_number': req['round_number'],
                    'weights': obj_to_pickle_string(my_weights),
                    'train_size': self.local_model.x_train.shape[0],
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'valid_loss': valid_loss,
                    'valid_accuracy': valid_accuracy,
                    'recv_request_time': w_update_start_time,
                    'training_end_time': end_time, # set to end_time to include evaluation overhead, 
                                                   # e_time otherwise
                    'sent_request_time': req['sent_request_time']
                }
                
                self.sio.emit('client_update', resp)
                self.logger.info("Round # {} finished. Sent Response to server.".format(req['round_number']))
                self.logger.debug("Receive update of bytes: {}".format(sys.getsizeof(req)))
                self.logger.debug("Send update of bytes: {}".format(sys.getsizeof(resp)))


        def on_stop_and_eval(*args):
            """Never called."""
            # TODO: Keep this?            
            req = args[0]
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weights(weights)
            test_loss, test_accuracy = self.local_model.evaluate()
            resp = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            self.sio.emit('client_eval', resp)

        self.sio.on('connect', on_connect)
        self.sio.on('error', on_connection_error)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('connection_error', on_connection_error)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)
    
    def intermittently_sleep(self, p=.1, low=10, high=100):
        """Never called."""
        # TODO: Keep this?       
        if (random.random() < p):
            time.sleep(random.randint(low, high))



if __name__ == "__main__":
    # Get the Configuration File, the Client # and Tier it belongs to
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    con = tf.ConfigProto()
    con.gpu_options.allow_growth=True   
    con.allow_soft_placement = True
    con.log_device_placement = False
    sess = tf.Session(config=con)

    client_num = int(sys.argv[1])
    tier = int(sys.argv[2])
    CustomLogger.show("FL-CLIENT-MAIN, Client {} tier {}".format(client_num, tier),
            "Using Configuration File: {}".format("default_config.py"))
    from default_config import config
    
    client = FederatedClient(config, client_num, tier)
