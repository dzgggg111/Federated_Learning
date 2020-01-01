#!/usr/bin/python3
import pickle
import uuid
from custom_logger import CustomLogger

from fl_models import global_models
import datasource
import tensorflow as tf

from keras import backend as K

import codecs # For pickling/unpickling
import numpy as np

from datetime import timedelta
import time
import sys
from threading import Lock
import csv
import os
# https://flask-socketio.readthedocs.io/en/latest/
from flask import *
from flask_socketio import *
import logging

logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# -------------------------------------------------

from numpy.random import seed
seed(int(time.time())) # Generate a new seed every time code is run
import random

# Federated Averaging algorithm with the server pulling from clients
class FLServer(object):

    def __init__(self, config):
        """
        Initializes FLServer's required variables.
        
        Does nothing, only sets all object attributes.
        """
        self.logger = CustomLogger(config["server"]["logging_level"],
                                "SERVER", config["server"]["log_file"])

        # SERVER ---------------------------
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, ping_timeout=300, ping_interval=120)
        self.host = config["server"]["host"]
        self.port = config["server"]["port"]
        self.logger.debug("Launching in {}:{}".format(self.host, self.port))
        # ----------------------------------
        
        # MODEL AND TRAINING_POLICIES --------------------------------
        self.global_model = global_models[config["server"]["model"]]()
        self.eval_model = global_models[config['server']['model']]()
        self.policy = config["server"]["policy"]
        self.total_tiers = config["clients"]["tiers"]
        self.total_clients = config["clients"]["clients_per_tier"] * self.total_tiers
        self.training_rounds = config["server"]["total_rounds"]
        self.clients_trained_per_round = config["server"]["clients_trained_per_round"]
        self.eval_every_n_rounds = config["server"]["eval_every_n_rounds"]
        self.training_batch_size = config["server"]["training_batch_size"]
        self.logger.info("Using Model: {}".format(type(self.global_model).__name__))
        self.logger.info("Use Policy: {}, Train for {} rounds ({} clients per round)".format(
                        self.policy, self.training_rounds, self.clients_trained_per_round))
        self.logger.info("Total {} Clients spread through {} Tiers ({} per tier)".format(
                        self.total_clients, self.total_tiers, 
                        config["clients"]["clients_per_tier"]))
        self.dead = False
        # ------------------------------------------------------------
        self.offset = 0
        # DATASET ------------------------------------------------------
        data_split = config["dataset"]["data_split"]
        if config["dataset"]["log_file"]:
            config["dataset"]["log_file"] = config["dataset"]["log_file"].format("server")
        dataset = datasource.available_datasets[config["dataset"]["name"]](config["dataset"])
        
        if config["dataset"]["iid"] :
            all_idxs = dataset.iid(data_split)
        else :
            if config["dataset"]["subtiered"]:
                all_idxs = dataset.non_iid_subtiered( 
                    classes_per_client=config["dataset"]["classes_per_client"], 
                    uniform=config["dataset"]["uniform"], repeat=config["dataset"]["repeats"],
                    tiers=config["clients"]["tiers"], subtiers=2,
                    clients_per_tier=config["clients"]["clients_per_tier"])
            else:
                all_idxs = dataset.non_iid(data_split,
                    classes_per_client=config["dataset"]["classes_per_client"],
                    uniform=config["dataset"]["uniform"],
                    repeats=config["dataset"]["repeats"])
        
            
        self.data_index, self.test_idxs = all_idxs
        self.total_num_updates = 0
        
        self.logger.debug("Generating Validation Data...")
        self.val_x, self.val_y = dataset.generate_data(
            dataset.x_test, dataset.y_test) # Validation Data
        self.tierwise_test_idxs, self.total_test_idxs = None, None # Not setting it up here since we need client idxs
        self.logger.info("Completed Generating Test Data by user index...")
        # --------------------------------------------------------------
        
        # TIME AND ACC RECORDS ----------
        self.lock = Lock()
        self.latest_round_lock = Lock()
        self.training_q_lock = Lock()
        self.weight_updates_lock = Lock()
        self.latest_round = {}
        self.ready_client_sids = []
        self.model_id = str(uuid.uuid4())
        self.results = []
        self.temp_dict = dict()
        self.results_pre = []
        self.tem_dict_pre = dict()
        self.current_client = 0
        self.round_start_time = 0
        self.training_start_time = 0
        # -------------------------------
        
        # TRAINING STATES --------------------
        self.current_round = -1 if config['server']['sync'] else 0# -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        # ------------------------------------

        # SOCKET IO MESSAGE HANDLING -----------------------------
        self.register_handles()
        self.config = config
        # --------------------------------------------------------

        self.tierwise_test_idxs, self.total_test_idxs = self.set_up_tierwise_test_idxs()

        self.waiting_to_train_q = []

        self.logger.info("Finished Initialization")
        

    def wait_for_stragglers(self, current_client, max_round_diff):

        if (max_round_diff <= -1 or 
            len(self.latest_round.keys()) <= 0 or
            current_client not in self.latest_round.keys()) :
            return False

        def get_slowest_client_round():
            self.latest_round_lock.acquire()
            slowest_round = min([latest_rnd for client, latest_rnd in self.latest_round.items()])
            self.latest_round_lock.release()
            return slowest_round

        slowest_client_round = get_slowest_client_round()
        curr_round = self.latest_round[current_client]
        self.logger.debug('Curr client rounds {},'
                              ' slowest {}'.format(curr_round, 
                                                   slowest_client_round))
        if (curr_round - slowest_client_round) > max_round_diff:
            self.logger.debug('Diff more than {},'
                              ' so sending to queue'.format(max_round_diff))
            return True

        self.logger.debug('Diff is less than {}, so should move on.'.format(max_round_diff))
        return False
    
        
    def save_result(self, csv_file, dict_data):
        """Prints all recorded times and accuracies to file in csv format"""
        csv_columns = dict_data[0].keys()
        print('csv_columns is ', csv_columns)
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)


    def set_up_tierwise_test_idxs(self):
        """Set up dictionary where key: tier number, 
           value: test idxs for the clients in that index
        """
        tierwise_test_idxs = {}
        total_test_idxs = []
        for _, tier, name in self.ready_client_sids:
            client_num = int(name.split("_")[-1]) # Client num is the same as dataset index
            idxs = self.test_idxs[client_num].tolist()
            tierwise_test_idxs.setdefault(int(tier), []).extend(idxs)
            total_test_idxs.extend(idxs)
        return tierwise_test_idxs, total_test_idxs
        
        
    def get_name_from_sid(self, sid):
        """Retrieves the corresponding client name of the sid"""
        for client_sid, _, name in self.ready_client_sids:
            if sid == client_sid:
                return name
        return "!!! Unknown Client !!!"
    
    
    def start(self):
        """Start the Server and listen for incoming connections."""
        self.logger.info("Server Started. Waiting for {} Clients...".format(self.total_clients))
        self.socketio.run(self.app, host=self.host, port=self.port,debug=False)
        self.logger.info("Stopped. Qutting...")
        
        
    def register_handles(self):
        """
        Registers all SOCKET IO handles.
        
        Socket protocols -> connect, reconnect, disconnect, error_default, 
        client_wake_up, client_update, client_ready, client_eval.
        """
            
        @self.socketio.on('connect')
        def handle_connect():
            """Logs when client initiated contact."""
            self.logger.info("Session started by client : {}".format(request.sid))
            
            # TODO: Remove Later - FOR DEBUGGING ONLY ------------------------------------
            # prettified_attr = self.logger.prettify(dir(request))
            # prettified_vars = self.logger.prettify(vars(request))
            # self.logger.debug("Connected Request Attributes: \n{}".format(prettified_attr))
            # self.logger.debug("Connected Request Variables: \n{}".format(prettified_vars))
            # -----------------------------------------------------------------------------

        @self.socketio.on('reconnect')
        def handle_reconnect():
            """Logs when a client disconnects and reconnects."""
            self.logger.info("Session reconnected by client : {}".format(request.sid))
            
            # TODO: Remove Later - FOR DEBUGGING ONLY --------------------------------------
            # prettified_attr = self.logger.prettify(dir(request))
            # prettified_vars = self.logger.prettify(vars(request))
            # self.logger.debug("Reconnected Request Attributes: \n{}".format(prettified_attr))
            # self.logger.debug("Reconnected Request Variables: \n{}".format(prettified_vars))
            # ------------------------------------------------------------------------------
            
        # @self.socketio.on_error_default
        def handle_error(e):
            """Catches and Logs all Server Socket Handling errors."""
            self.logger.err("Error from {}".format(request.sid))
            self.logger.err(request.event["message"])
            self.logger.err("args: {}".format(request.event["args"]))
            self.logger.err("Error: {}".format(e))
            self.logger.debug("Error type: {}".format(type(e)))
            
            # TODO: Remove Later - FOR DEBUGGING ONLY ------------------------------
            # prettified_attr = self.logger.prettify(dir(e))
            # prettified_vars = self.logger.prettify(vars(e))
            # self.logger.debug("Error Attributes {}".format(prettified_attr))
            # self.logger.debug("Error Variables {}".format(prettified_vars))
            # prettified_attr = self.logger.prettify(dir(request))
            # prettified_vars = self.logger.prettify(vars(request))
            # self.logger.debug("Error Request Attributes: {}".format(prettified_attr))
            # self.logger.debug("Error Request Variables: {}".format(prettified_vars))
            # -----------------------------------------------------------------------
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Updates list of available clients when a client session disconnects."""
            self.logger.info("Client {} disconnected".format(self.get_name_from_sid(request.sid)))

            new_list = [ (sid, tier, name) for sid, tier, name in self.ready_client_sids if sid != request.sid ]
            self.ready_client_sids = new_list
            self.logger.debug('Removed from available clients.')

            self.latest_round_lock.acquire()
            self.latest_round.pop(request.sid, None)
            self.latest_round_lock.release()
            self.logger.debug('Removed from latest round')

            self.training_q_lock.acquire()
            try :
                self.training_q.remove(request.sid)
            except:
                self.logger.debug('Did not remove anything from training q')
            self.training_q_lock.release()
            self.logger.debug('Removed from training q')

            self.logger.debug("Clients still available: {}".format(self.ready_client_sids))
            exit(1)
            self.async_train_next_round(request.sid)


        @self.socketio.on('client_wake_up')
        def handle_wake_up(data):
            """
            Replies with the data indices, model and training parameters to a client.
            
            Basically the second phase of the three-way handshake.
            On being called, the training dataset is configured for the calling client
            by selecting the appropriate user_index of the full training dataset.
            Sends the model, the training indices, the batch size and number of epochs
            to run. Also updates the total connected clients count.
            """
            self.logger.debug("client_wake_up called from {}".format(request.sid))
            self.logger.info("New Client is available. Name of Client # {}. "
                             "Sending model and dataset..".format(data['name']))
            
            # If more clients are available than dataset indices, reuse dataset
            # TODO: Is this the correct thing to do?
            # JUST USE THE CLIENT'S NAME (which contains it's number) AS
            # THE INDEX. On the Client side, ensure that no two clients have the same index
            
            dataset_idx_for_client = int(data['name'].split("_")[-1]) # the last part of the name
            dataset_idx_for_client %= self.config["dataset"]["num_users"] # Repeat dataset if too many clients
            dataset_idx_for_client = dataset_idx_for_client + self.offset*5
            self.logger.debug("Using dataset index {} for it".format(dataset_idx_for_client))
            emit('init', {
                    'model_json': self.global_model.model.to_json(),
                    'model_id': self.model_id,
                    'dataset': self.config['dataset']['name'],
                    'epoch_per_round': 1,
                    'test_indices': self.test_idxs[dataset_idx_for_client].tolist(),
                    'batch_size': self.training_batch_size,
                    'indices':self.data_index[dataset_idx_for_client].tolist()
                })
            self.current_client += 1


        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            """
            Adds the calling client to the list of ready workers and starts training if possible.
            
            The last phase of the three-way handshake. The client which calls this function effectively
            says 'I am ready to train my model' and so will be added to the pool of available-to-train
            workers. Starts training if sufficient number of clients are available in the pool.
            """
            self.logger.debug("Client {} called \'client_ready\'. Recieved data: {}".format(request.sid, data))
            self.logger.info("Client {} ready for training".format(data['client_name']))
            self.logger.debug("Client Details ->")
            self.logger.debug("\tName: {}".format(data['client_name']))
            self.logger.debug("\tTier: {}".format(data['client_tier']))
            self.logger.debug("\tHash: {}".format(request.sid))
            self.logger.debug("\tHost: {}:{}".format(request.environ['REMOTE_ADDR'], 
                                                     request.environ['REMOTE_PORT']))
            
            self.lock.acquire()
            self.ready_client_sids.append((request.sid, data['client_tier'], data['client_name']))
            self.lock.release()
            self.start_time = time.time() 
            self.logger.info("Currently ready clients: {}".format(len(self.ready_client_sids)))
            self.logger.debug(self.logger.prettify(self.ready_client_sids))
            
            if not self.config['server']['sync']:
                self.async_train_next_round(request.sid)
            elif len(self.ready_client_sids) >= self.total_clients and self.current_round == -1:
                self.logger.info("# of Clients is enough, starting training...")
                self.tierwise_test_idxs, self.total_test_idxs = self.set_up_tierwise_test_idxs()
                self.train_next_round


        @self.socketio.on('client_update')
        def handle_client_update(data):
            """
            Receives the trained weights from clients and updates the local model.
            Also evaluates & records test/train accuracies and losses.
            
            Called when a client has finished one round of training. It records the 
            training time breakdown for the clients as well as the full round. Adds the 
            weights to an aggregator matrix. If all clients are done with the training,
            it updates the local model weight by aggregating the weights from the weight matrix.
            It then starts a new round, or stops and prints out the data if last round.
            """

            def update_latest_round(for_client, to):
                self.latest_round_lock.acquire()
                if for_client not in self.latest_round.keys(): 
                    self.latest_round[for_client] = 0
                self.latest_round[for_client] += 1
                self.latest_round_lock.release()
                return

            if self.dead:
                self.logger.info("Server is dead, so no updates")
                return 0
            client_name = self.get_name_from_sid(request.sid)
            self.logger.debug("Recieved client weight updates from {}".format(client_name))
            update_latest_round(for_client=request.sid, to=data['round_number'])

            # Discarding outdated updates
            if self.config['server']['sync'] and data['round_number'] != self.current_round:
                self.logger.debug("Outdated update, so ignoring it.")
                return
            
            # TIMING THE TRAINING STEP PER CLIENT ---------------------------------
            
            client_reached_server_time = time.time()
            server_sent_request_time = data['sent_request_time']
            client_recv_request_time = data['recv_request_time']
            client_training_end_time = data['training_end_time']
            client_recv_delay = client_recv_request_time - server_sent_request_time
            training_time = client_training_end_time - client_recv_request_time
            server_recv_delay = client_reached_server_time - client_training_end_time
            # self.logger.debug("Training Time Breakdown ->")
            # self.logger.debug("\tServer -> Client: {}s".format(client_recv_delay))
            # self.logger.debug("\tTraining: {}s".format(training_time))
            # self.logger.debug("\tClient -> Server: {}s".format(server_recv_delay))
            # self.logger.debug("\tTotal Time: {}s".format(
            #    (client_reached_server_time - server_sent_request_time)))
            # ------------------------------------------------------------------------
            
            # Update Weights - Need locks to avoind multiple clients trying
            # to update weight matrix at the same time
            self.lock.acquire()
            self.current_round_client_updates += [data]
            self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
            clients_responded = len(self.current_round_client_updates)
            self.logger.debug("{} clients sent their weights this round".format(clients_responded))
            self.lock.release()

            # Move on if not enough clients are available            
            if self.config['server']['sync'] and clients_responded < 5:
                self.logger.debug("Not enough clients for weight updates")
                return

            self.lock.acquire()
            if self.config['server']['sync']:
                self.logger.info("Round {}: Recieved trained weights from all clients. "
                             "Updating weights.....".format(self.current_round))
                self.update_weights()
            else :
                self.total_num_updates += 1
                self.logger.info("Round {}, Total Updates {}: "
                                 "Updating weights.....".format(self.current_round,
                                                                self.total_num_updates))
                self.async_weight_updates()
                if self.total_num_updates % 5 == 0:
                    self.current_round += 1
                    self.current_round_client_updates = []
                    self.logger.info("Round incremented to {}".format(self.current_round))
                    self.logger.info("Testing for client {}".format(request.sid))
                    self.lock.release()
                    self.record_test_time()
            self.record_train_time()

            try:
                self.lock.release()
            except:
                pass
           
            if self.config['server']['sync']:
                if self.current_round >= self.training_rounds:
                    # TODO: Does not exit itself. WHY???
                    self.save_result('train_valid_old.csv', self.results)
                    self.save_result('test_old.csv', self.results_pre)
                    self.logger.info("Completed {} rounds. Total Time taken: {}s".format(
                        self.current_round, time.time() - self.training_start_time))
                    self.logger.info("Full Training Complete. Committing Suicide....")
                    self.logger.debug("Vale Mundi!")
                    time.sleep(5) # Pause for dramatic effect
                    exit(0)        # then commit suicide
                elif self.current_round % self.eval_every_n_rounds == 0:
                    self.record_test_time()
                self.train_next_round()
            else :
                self.lock.acquire()
                if self.current_round >= self.training_rounds and not self.dead:
                    self.dead = True
                    self.logger.info("\n\n\n\t\t\t\tDIE!!!!")
                    self.save_result('train_valid_old.csv', self.results)
                    self.save_result('test_old.csv', self.results_pre)
                    self.logger.info("Completed {} rounds. Total Time taken: {}s".format(
                        self.current_round, time.time() - self.training_start_time))
                    self.logger.info("Full Training Complete. Committing Suicide....")
                    self.logger.debug("Vale Mundi!")
                    self.lock.release()
                    time.sleep(5) # Pause for dramatic effect
                    exit(0)
                if not self.dead:
                    self.async_train_next_round(request.sid)    
                try:
                    self.lock.release()
                except:
                    pass


        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            """Never Used."""
            # TODO: Remove this?
            if self.eval_client_updates is None:
                return
            print("handle client_eval", request.sid)
            print("eval_resp", data)
            self.eval_client_updates += [data]

            # tolerate 30% unresponsive clients
            if len(self.eval_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * 1.0:
                aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                );
                print("\naggr_test_loss", aggr_test_loss)
                print("aggr_test_accuracy", aggr_test_accuracy)

                self.tem_dict_pre['round'] = self.current_round
                self.tem_dict_pre['test loss'] = aggr_test_loss
                self.tem_dict_pre['test accuracy'] = aggr_test_accuracy
                self.results_pre.append(self.tem_dict_pre)
                self.tem_dict_pre = dict()

                print('self.results ', self.results)
                self.save_result('train_valid_old.csv', self.results)
                self.save_result('test_old.csv', self.results_pre)
                print("== done ==")
                self.eval_client_updates = None  # special value, forbid evaling again


    def record_train_time(self):
        """Records total time, full training set loss/accuracy of current round"""
        self.temp_dict['round'] = self.current_round
        self.temp_dict['time cost'] = time.time() - self.start_time
        self.temp_dict['train_loss'] = self.global_model.prev_train_loss[-1]
        self.temp_dict['train_acc'] = self.global_model.prev_train_acc[-1]
        self.logger.info("Round {} ->".format(self.current_round))
        self.logger.info("\tTime: {}s".format(self.temp_dict['time cost']))
        self.logger.info("\tTrain Accuracy/Loss: {:.2f}/{:.3f}".format(
            self.temp_dict['train_acc'], self.temp_dict['train_loss']))
        self.results.append(self.temp_dict)
        self.temp_dict = dict()


    def record_test_time(self):
        """Evaluates and records test loss/accuracy of current round"""
        self.offset = self.offset + 1
        eval_weights = self.global_model.model.get_weights()
        self.eval_model.current_weights = eval_weights
        self.tem_dict_pre = dict()
        with self.eval_model.graph.as_default():
            self.eval_model.set_weights()
            self.logger.info("Logging Test accuracies...")
            '''
            for tier_num in range(1, self.total_tiers+1):
                test_idxs = self.tierwise_test_idxs[tier_num]
                x_val = self.val_x[test_idxs]
                y_val = self.val_y[test_idxs]
                test_loss, test_accy = self.global_model.evaluate(x_val, y_val)
                self.tem_dict_pre['test loss_' + str(tier_num)] = test_loss
                self.tem_dict_pre['test accuracy_' + str(tier_num)] = test_accy
                self.tem_dict_pre['round'] = self.current_round
                self.logger.info("Round {}: Using Tier {}'s test set, loss {}, acc {}".format(
                    self.current_round, tier_num, test_loss, test_accy))
                self.logger.debug("Getting F1 scores....")
                f1_scores = self.get_f1_scores(x_val, y_val)
                self.logger.info("{} F1 {} {}".format(self.current_round, tier_num, f1_scores))
            '''
            x_val, y_val = self.val_x, self.val_y
            loss, acc = self.eval_model.evaluate(x_val, y_val)
            self.tem_dict_pre['total_acc'] = acc
            self.tem_dict_pre['total_loss'] = loss
            self.logger.info("Round {}: Total acc/loss: {:.4f}/{:.3f}".format(self.current_round, acc, loss))
            # self.logger.info("{} F1 {} {}".format(self.current_round, 'all', f1_scores))
        self.results_pre.append(self.tem_dict_pre)
        self.logger.info("...Done")


    def record_test_time_old(self):
        """Evaluates and records test loss/accuracy of current round"""
        
        with self.global_model.graph.as_default():
            self.global_model.set_weights()
            self.logger.info("Logging Test accuracies...")
            num_users = len(self.test_idxs)
            for user_idx in range(num_users):
                x_val = self.val_x[self.test_idxs[user_idx]]
                y_val = self.val_y[self.test_idxs[user_idx]]
                test_loss, test_accy = self.global_model.evaluate(x_val, y_val)
                self.tem_dict_pre['test loss_' + str(user_idx)] = test_loss
                self.tem_dict_pre['test accuracy_' + str(user_idx)] = test_accy
                self.tem_dict_pre['round'] = self.current_round
                self.logger.info("Round {}: Using client_{}'s test set, loss {}, acc {}".format(
                    self.current_round, user_idx, test_loss, test_accy))
        self.results_pre.append(self.tem_dict_pre)
        self.logger.info("...Done")
        self.tem_dict_pre = dict()


    def update_weights(self):
        """Updates the local model's weights using the client's aggregate weights"""
        self.global_model.update_weights(
            [x['weights'] for x in self.current_round_client_updates],
            [x['train_size'] for x in self.current_round_client_updates],
        )
        with self.global_model.graph.as_default():
            self.global_model.set_weights()
        aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
            [x['train_loss'] for x in self.current_round_client_updates],
            [x['train_accuracy'] for x in self.current_round_client_updates],
            [x['train_size'] for x in self.current_round_client_updates],
            self.current_round
        )
        self.global_model.prev_train_loss.append(aggr_train_loss)
        self.global_model.prev_train_acc.append(aggr_train_accuracy)


    def async_weight_updates(self):
        """Updates the local model's weights using the client's aggregate weights"""
        self.weight_updates_lock.acquire()
        self.global_model.update_weights(
            [x['weights'] for x in self.current_round_client_updates],
            [x['train_size'] for x in self.current_round_client_updates],
        )
        with self.global_model.graph.as_default():
            self.global_model.set_weights()
        aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
            [x['train_loss'] for x in self.current_round_client_updates],
            [x['train_accuracy'] for x in self.current_round_client_updates],
            [x['train_size'] for x in self.current_round_client_updates],
            self.current_round
        )
        self.global_model.prev_train_loss.append(aggr_train_loss)
        self.global_model.prev_train_acc.append(aggr_train_accuracy)
        self.weight_updates_lock.release()
        return


    def async_train_next_round(self, sid):
        self.weight_updates_lock.acquire()
        pickled = obj_to_pickle_string(self.global_model.current_weights)
        self.weight_updates_lock.release()

        self.training_q_lock.acquire()
        self.waiting_to_train_q.append(sid)
        train_this_round_q = []
        for sid in self.waiting_to_train_q:
             if not self.wait_for_stragglers(current_client=sid, 
                                         max_round_diff=self.config['server']['async_bound']):
                 train_this_round_q.append(sid)
        for sid in train_this_round_q:
             self.waiting_to_train_q.remove(sid)
        self.training_q_lock.release()

        for sid in train_this_round_q:
            self.logger.debug("Requesting updates from {}".format(sid))
            emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'run_validation': False,
                    'run_test': False,
                    'current_weights': pickled,
                    'weights_format': 'pickle',
                    'sent_request_time': time.time()
                }, room=sid)
        self.logger.debug('Remaining clients in queue- {}'.format(self.waiting_to_train_q))


    def train_next_round(self):
        """
        Train One Round. Randomly select clients based on policy and configurations
        and request updates from selected clients.
        """
        self.logger.debug("Training One Round")
        self.current_round += 1
        self.current_round_client_updates = [] # Weight Buffers
        
        # POLICY ->
        self.logger.info("Starting Training Round # {}".format(self.current_round))
        if self.policy:
            random_tier = int(np.random.choice(
                range(1, self.total_tiers + 1), 1, p=self.config['server']['tier_distribution']))
            self.logger.info("Using policy, selected Tier {} randomly".format(random_tier))
            sids_pool = [ (sid, tier, name) for sid, tier, name in self.ready_client_sids if int(tier) == random_tier ]
        else:
            sids_pool = self.ready_client_sids
        self.logger.debug("Selecting clients randomly from pool: {}".format(sids_pool))
        client_sids_selected = random.sample(sids_pool, 5)
        selected_client_names = [ name for _, _, name in client_sids_selected ]
        self.logger.info("Selected clients: {}".format(selected_client_names))
        
        if self.current_round == 0: # Starting Training so log time
            self.training_start_time = time.time()
        for rid, tier, client_name in client_sids_selected:
            pickled = obj_to_pickle_string(self.global_model.current_weights)
            self.logger.debug("Requesting updates from {}".format(client_name))
            emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'run_validation': False,
                    'run_test': False,
                    'current_weights': pickled,
                    'weights_format': 'pickle',
                    'sent_request_time': time.time()
                }, room=rid)
        self.logger.debug("Done sending update requests, waiting for responses...")
        self.start_time = time.time()
        
        
    def get_f1_scores(self, test_x, test_y):
        """Calculate the f1 score for every class"""
        with self.global_model.graph.as_default():
            pred_y = self.global_model.predict(test_x) # Get prediction probabilities
        pred_y = np.argmax(pred_y, axis=1)             # Argmax vector for predictions
        actual_y = np.argmax(test_y, axis=1)           # Argmax vector for actual values
        num_classes = int(np.amax(actual_y)) + 1       # [0-9] class categories 
        f1_scores = [0.0] * num_classes
        zipped = list(zip(pred_y, actual_y))     # Make list of tuples [(predicted, actual), ...]
                                                 # Easier to filter this way
        for class_num in range(num_classes): # For every class 0-9 get the f1 score
            TP = len(list(filter( lambda x: x[0] == class_num and x[1] == class_num, zipped )))
            FP = len(list(filter( lambda x: x[0] == class_num and x[1] != class_num, zipped )))
            FN = len(list(filter( lambda x: x[0] != class_num and x[1] == class_num, zipped )))
            # True Negative is not required for F1 scores
            # TN = len(list(filter( lambda x: x[0] != class_num and x[1] != class_num, zipped )))
            
            # No Need for recall and precision
            f1_scores[class_num] = (2 * TP) / ( (2 * TP) + FP + FN ) # Simplified formula for F1 score
            
        return f1_scores


    def stop_and_eval(self):
        """Never used."""
        # TODO: Remove this?
        self.end_time = time.time()
        # print("Time cost : ", self.end_time - self.start_time)
        self.eval_client_updates = []
        for rid, rid_tier, name in self.ready_client_sids:
            emit('stop_and_eval', {
                    'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle'
                }, room=rid)
            self.socketio.sleep(0)


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)


if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    # config_file = "default_config.json" if not len(sys.argv) >= 2 else sys.argv[1]
    # with open(config_file) as json_file:
    #     config = json.load(json_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    con = tf.ConfigProto()
    con.gpu_options.allow_growth=True   
    con.allow_soft_placement = True
    sess = tf.Session(config=con)




    CustomLogger.show("FL-SERVER-MAIN", "Using Configuration File: {}".format("default_config.py"))
    from default_config import config
    server = FLServer(config)
    server.start()
