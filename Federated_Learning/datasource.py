#!/usr/bin/python3
import numpy as np
import json
import keras
import random
from keras.datasets import mnist
from custom_logger import CustomLogger
from keras import backend as K
from keras.datasets import cifar10
import time

# TODO: Fix MNIST - fix iid and test non-iid
# TODO: Remove datasplit altogether
class DataSource(object):
    
    def __init__(self):
        raise NotImplementedError()
    
    def generate_data(self, x, y):
        raise NotImplementedError
    
    def iid(self, data_split):
        raise NotImplementedError
    
    def non_iid(self, data_split):
        raise NotImplementedError
    
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        raise NotImplementedError()
    
    def sample_single_non_iid(self, weight=None):
        raise NotImplementedError()


class Mnist(DataSource):

    def __init__(self, config):
        
        self.log = CustomLogger(logging_level=config["logging_level"], 
                                logging_class="MNIST DATASOURCE",
                                output_to_file=config["log_file"])
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.num_users = config["num_users"]
        self.num_shards = config["num_shards"]
        self.classes = np.unique(self.y_train)
        
        # DEBUGGING
        self.log.debug("Initialized ->")
        self.log.debug("\tTraining Images : {}".format(len(self.x_train)))
        self.log.debug("\tTest Images: {}".format(len(self.x_test)))
        self.log.debug("\tClasses: {}".format(self.classes))


    def iid(self, data_split=[1.0, 0]):
        """
        Sample I.I.D. client data from MNIST dataset
        :param dataset:
        :param num_users:
        :return: Two dictionaries (train/test) of iamge indices keyed by user num
        """
        
        self.log.debug("Generating IID indices for {} users...".format(self.num_users))
        self.log.debug("Train/Test Ratio: {}".format(data_split))
        
        train_items_per_user = int(len(self.x_train) / self.num_users)
        test_items_per_user  =  int(len(self.x_test) / self.num_users)
        dict_users_train, dict_users_test = {}, {}
        train_idxs, test_idxs = [i for i in range(len(self.x_train))], [i for i in range(len(self.x_test))]
        for i in range(self.num_users):
            dict_users_train[i] = np.random.choice(train_idxs, train_items_per_user, replace=False)
            train_idxs = list(set(train_idxs) - set(dict_users_train[i]))
            dict_users_test[i] = np.random.choice(test_idxs, test_items_per_user, replace=False)
            test_idxs = list(set(test_idxs) - set(dict_users_test[i]))

        self.log.debug("# of users: {}, train/test images per user: {}/{}".format(self.num_users,
                                                                train_items_per_user, test_items_per_user))
        self.log.info("IID Dataset generated")

        return dict_users_train, dict_users_test


    def non_iid(self, data_split=[1.0, 0.0], classes_per_client=5):
        """
        Generates Non-IID data. Returns test and train dictionaries of 
        client number to image index. Splits up the datasets by classes
        and allocates images of specific classes per client.
        """
        import random # FYI: Importing in functions is a bad idea

        total_train_imgs, total_test_imgs = 50000, 10000
        train_imgs_perclient, test_imgs_perclient = int(50000/self.num_users), int(10000/self.num_users)
        num_shards = self.num_users * classes_per_client
        y_train, y_test = self.y_train.flatten(), self.y_test.flatten()

        # Key - Class Number (0-9), Value: index list of 
        # all images belonging to that class
        train_indices_by_class = {}
        test_indices_by_class = {}

        # Train dataset
        for img_index, class_num in enumerate(y_train):
            train_indices_by_class.setdefault(class_num, []).append(img_index)

        for img_index, class_num in enumerate(y_test):
            test_indices_by_class.setdefault(class_num, []).append(img_index)

        # Number of images per class per client
        num_test_imgs_perclass = int(total_test_imgs / num_shards)
        num_train_imgs_perclass = int(total_train_imgs / num_shards)

        # Key - client number, Value - List of image indices for that client
        train_idx_per_client = {}
        test_idx_per_client = {}

        classes_avail = list(self.classes)
        for client_num in range(self.num_users):
            # For every client, randomly choose a class and assign the images
            # of those classes to the clients, until it it reaches max number of imgs
            # required

            train_idx_per_client[client_num] = []
            test_idx_per_client[client_num] = []
            while len(train_idx_per_client[client_num]) < train_imgs_perclient:
                random_class = random.sample(classes_avail, k=1)[-1]

                train_indices = train_indices_by_class[random_class][:num_train_imgs_perclass]
                train_indices_by_class[random_class] = train_indices_by_class[random_class][num_train_imgs_perclass:]
                train_idx_per_client[client_num].extend(train_indices)

                test_indices = test_indices_by_class[random_class][:num_test_imgs_perclass]
                test_indices_by_class[random_class] = test_indices_by_class[random_class][num_test_imgs_perclass:]
                test_idx_per_client[client_num].extend(test_indices)

                if len(test_indices_by_class[random_class]) == 0:
                    classes_avail.remove(random_class)
                
        for client_num in range(self.num_users):
            train_idx_per_client[client_num] = np.array(train_idx_per_client[client_num])
            test_idx_per_client[client_num] = np.array(test_idx_per_client[client_num])

        self.log.debug("Generated Non-iid images."
                       " {} clients, {} classes per"
                       " client ->".format(self.num_users, classes_per_client))
        self.log.debug("{}/{} train/test images per client".format(
            num_train_imgs_perclass * classes_per_client, 
            num_test_imgs_perclass * classes_per_client
        ))
        
        return train_idx_per_client, test_idx_per_client


    def non_iid_old(self, data_split):
        """
        Sample non-I.I.D client data from MNIST dataset
        :param dataset:
        :param num_users:
        :return: Two dictionaries (train/test) of iamge indices keyed by user num
        """
        self.log.debug("Generating Non-IID indices for {} users...".format(self.num_users))
        self.log.debug("Train/Test Ratio: {}, Shards: {}".format(data_split, self.num_shards))
        
        shards_per_user = int(self.num_shards / self.num_users)
        num_images_per_shard = int(len(self.y_train) / self.num_shards)
        idx_shard = [i for i in range(self.num_shards)]
        dict_users_train = {i: np.array([], dtype='int64') for i in range(self.num_users)}
        dict_users_test = {i: np.array([], dtype='int64') for i in range(self.num_users)}
        idxs = self.y_train.argsort()

        for i in range(self.num_users):
            rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            
            self.log.debug("Using shard/s {} for User {}".format(rand_set, i))
            
            for rand in rand_set:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], 
                     idxs[ 
                        rand * num_images_per_shard : 
                        rand * num_images_per_shard + int(num_images_per_shard * data_split[0])
                        ]
                    ), axis=0)
                dict_users_test[i] = np.concatenate(
                    (dict_users_test[i], 
                     idxs[
                        rand * num_images_per_shard + int(num_images_per_shard * data_split[0]) :
                        (rand + 1) * num_images_per_shard
                        ]
                    ), axis=0)
                
        # DEBUG LOGGING
        train_set_count = len(dict_users_train[0])
        test_set_count = len(dict_users_test[0])
        self.log.debug("Generated non-IID training/test dataset.")
        self.log.debug("Train/Test size: {}/{} per user, {} users".format(
                        train_set_count, test_set_count, self.num_users))
        self.log.info("Non-IID Dataset generated")

        return dict_users_train, dict_users_test

    def generate_data(self, x, y):
        """
        Formats MNIST Data for use in Tensorflow.
        Reshapes, converts to float32 and normalizes images, and generates 
        one-hot encoded labels
        """
        self.log.debug("Formatting data for use in tensorflow model")
        self.log.debug("Initial Data format ->")
        self.log.debug("\tImage matrix dimensions: {}".format(x.shape))
        self.log.debug("\tLabel Matrix Dimensions: {}".format(y.shape))
        
        # input image dimensions
        img_rows, img_cols = 28, 28
        if K.image_data_format() == 'channels_first':
            x = x.reshape(x.shape[0], 1, img_rows, img_cols)
        else:
            x = x.reshape(x.shape[0], img_rows, img_cols, 1)
        x = x.astype('float32')
        x /= 255

        # convert class vectors to binary class matrices
        y = keras.utils.to_categorical(y, self.classes.shape[0])
        self.log.debug("Processed Data format ->")
        self.log.debug("\tImage matrix dimensions: {}".format(x.shape))
        self.log.debug("\tLabel Matrix Dimensions: {}".format(y.shape))
        
        return x, y
    

class Cifar10(DataSource):

    def __init__(self, config):
        
        self.log = CustomLogger(logging_level=config["logging_level"], 
                                logging_class="CIFAR10 DATASOURCE",
                                output_to_file=config["log_file"])
        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.y_train = np.array(self.y_train).flatten()
        self.y_test = np.array(self.y_test).flatten()
        self.num_users = config["num_users"]
        self.num_shards = config["num_shards"]
        self.classes = np.unique(self.y_train)
        # DEBUGGING
        self.log.debug("Initialized ->")
        self.log.debug("\tTraining Images : {}".format(len(self.x_train)))
        self.log.debug("\tTest Images: {}".format(len(self.x_test)))
        self.log.debug("\tClasses: {}".format(self.classes))
        
        
    def generate_data(self, x, y):
        """
        Formats CIFAR10 Data for use in Tensorflow.
        Converts to float32 and normalizes images, and generates 
        one-hot encoded labels
        """
        
        x = x.astype('float32')
        x /= 255

        # convert class vectors to binary class matrices
        y = keras.utils.to_categorical(y, self.classes.shape[0])
        self.log.debug("Generated Data format -> {}, {}".format(x.shape, y.shape))

        return x, y


    def get_classwise_indices(self, y):
        indices_by_class = {}
        for img_index, class_num in enumerate(y):
            indices_by_class.setdefault(class_num, []).append(img_index)
        return indices_by_class


    def non_iid(self, data_split=[1.0, 0.0], classes_per_client=5, uniform=False, repeats=0):
        """
        Generates Non-IID data. Returns test and train dictionaries of 
        client number to image index. Splits up the datasets by classes
        and allocates images of specific classes per client.
        """
        total_train_imgs, total_test_imgs = 50000, 10000
        num_shards = self.num_users * classes_per_client
        y_train, y_test = self.y_train.flatten(), self.y_test.flatten()

        # Key - Class Number (0-9), Value: index list of 
        # all images belonging to that class
        train_indices_by_class, train_common_indices_by_class = {}, {}
        test_indices_by_class = {}

        # Train dataset
        for img_index, class_num in enumerate(y_train):
            train_indices_by_class.setdefault(class_num, []).append(img_index)
            train_common_indices_by_class.setdefault(class_num, []).append(img_index)
            
        for img_index, class_num in enumerate(y_test):
            test_indices_by_class.setdefault(class_num, []).append(img_index)

        # Number of images per class per client
        num_test_imgs_perclass = int(total_test_imgs / num_shards)
        num_train_imgs_perclass = int(total_train_imgs / num_shards)

        # Key - client number, Value - List of image indices for that client
        train_idx_per_client = {}
        test_idx_per_client = {}

        classes_for_client = None
        while classes_for_client is None:
            classes_for_client = get_classes_per_client_dict(self.num_users, 
                                    classes_per_client, uniform, len(self.classes), total_train_imgs)
        
        for client_num in range(self.num_users):

            classes_assigned = classes_for_client[client_num]
            train_idx_per_client[client_num] = []
            test_idx_per_client[client_num] = []

            for class_assigned in classes_assigned:
                if not uniform:                
                    train_indices = train_indices_by_class[class_assigned][:num_train_imgs_perclass]
                    test_indices = test_indices_by_class[class_assigned][:num_test_imgs_perclass]
                    test_indices_by_class[class_assigned] = test_indices_by_class[class_assigned][num_test_imgs_perclass:]
                    train_indices_by_class[class_assigned] = train_indices_by_class[class_assigned][num_train_imgs_perclass:]
                else :
                    train_indices = np.random.choice(train_indices_by_class[class_assigned], num_train_imgs_perclass, replace=False)
                    leftover_train_indices = list(set(train_indices_by_class[class_assigned]) - set(train_indices))
                    train_indices_by_class[class_assigned] = leftover_train_indices
                    test_indices = np.random.choice(test_indices_by_class[class_assigned], num_test_imgs_perclass, replace=False)
                    leftover_test_indices = list(set(test_indices_by_class[class_assigned]) - set(test_indices))
                    test_indices_by_class[class_assigned] = leftover_test_indices

                remaining = list(set(train_common_indices_by_class[class_assigned]) - set(train_indices))
                num_repeat_choices = min(len(remaining), num_train_imgs_perclass * repeats)
                train_indices = np.append(train_indices, np.random.choice(
                    remaining, num_repeat_choices, replace=False)).astype(int)
                train_idx_per_client[client_num].extend(train_indices)
                test_idx_per_client[client_num].extend(test_indices)
        
        for client_num in range(self.num_users):
            train_idx_per_client[client_num] = np.array(train_idx_per_client[client_num])
            test_idx_per_client[client_num] = np.array(test_idx_per_client[client_num])

        self.log.debug("Generated Non-iid images."
                       " {} clients, {} classes per"
                       " client ->".format(self.num_users, classes_per_client))
        self.log.debug("{}/{} train/test images per client".format(
            num_train_imgs_perclass * classes_per_client * (1 + repeats), 
            num_test_imgs_perclass * classes_per_client
        ))
        
        return train_idx_per_client, test_idx_per_client


    def split_indxs_between_clients(self, dataset_per_class_indxs, num_clients=10, classes_per_client=5, uniform=True):

        num_classes = len(dataset_per_class_indxs.keys())
        num_subtiers = 2
        proportion = 0.5

        total_imgs = 0
        for class_num, indxs in dataset_per_class_indxs.items():
            total_imgs += len(indxs)

        num_imgs_per_client = total_imgs // num_clients
        num_imgs_per_client_per_class = num_imgs_per_client // classes_per_client

        classes_for_client = None
        while classes_for_client is None:
            classes_for_client = get_classes_per_client_dict(num_clients, 
                                    classes_per_client, uniform, num_classes, total_imgs)

        indxs_per_client = {}
        for client_num in range(num_clients):
            # print('Using classes {} for client {}'.format(classes_for_client[client_num], client_num))

            classes_assigned = classes_for_client[client_num]
            for class_assigned in classes_assigned:
                if not uniform:
                    selected_indxs = dataset_per_class_indxs[class_assigned][:num_imgs_per_client_per_class]
                    dataset_per_class_indxs[class_assigned] = dataset_per_class_indxs[class_assigned][num_imgs_per_client_per_class:]
                else :
                    selected_indxs = np.random.choice(dataset_per_class_indxs[class_assigned], num_imgs_per_client_per_class, replace=False)
                    leftover_indices = list(set(dataset_per_class_indxs[class_assigned]) - set(selected_indxs))
                    dataset_per_class_indxs[class_assigned] = leftover_indices

                    if client_num // ( num_clients // num_subtiers ) == 1:
                        indxs_from_source_client = np.random.choice(selected_indxs, int(len(selected_indxs) * proportion), replace=False)
                        selected_indxs = list(set(selected_indxs) - set(indxs_from_source_client))
                        target_client_num = client_num - ( num_clients // num_subtiers )
                        indxs_per_client.setdefault(target_client_num, []).extend(indxs_from_source_client)

                indxs_per_client.setdefault(client_num, []).extend(selected_indxs)

            # print("Client {} uses {} imgs".format(client_num, len(indxs_per_client[client_num])))
        
        """
        for c in range(num_clients):
            print("Client {} uses {} imgs".format(c, len(indxs_per_client[c])))
        """

        return indxs_per_client, classes_per_client


    def non_iid_subtiered(self, classes_per_client=10, uniform=True, repeat=0, tiers=5, subtiers=2, clients_per_tier=10):
        """
        Generates Non-IID data, but with tiers. 
        """
        y_train, y_test = self.y_train.flatten(), self.y_test.flatten()
        train_indices_by_class = self.get_classwise_indices(y_train)
        test_indices_by_class = self.get_classwise_indices(y_test)

        # TODO: Remove Hard-coded class index here
        total_imgs_per_class = len(train_indices_by_class[0])
        imgs_per_tier_per_class = total_imgs_per_class // tiers

        train_indices_tier_by_class = {}
        for tier_num in range(tiers):
            this_tier_classes = {}
            for class_num in train_indices_by_class.keys():
                indxs = train_indices_by_class[class_num]
                selected_indxs = np.random.choice(indxs, imgs_per_tier_per_class, replace=False)
                this_tier_classes[class_num] = selected_indxs
                remaining = list(set(indxs) - set(selected_indxs))
                train_indices_by_class[class_num] = remaining

            train_indices_tier_by_class[tier_num+1] = this_tier_classes
        
        """
        for tier_num in range(tiers):
            clses = train_indices_tier_by_class[tier_num+1]
            for cls, indxs in clses.items():
                print("{} - {} - {}".format(tier_num+1, cls, len(indxs)))
        """

        train_indices_by_client, test_indices_by_client = {}, {}
        client_num = 0
        for tier_num in range(tiers):
            train_client_to_indxs, classes_per_client = self.split_indxs_between_clients(
                                            num_clients=clients_per_tier, uniform=uniform, classes_per_client=5,
                                            dataset_per_class_indxs=train_indices_tier_by_class[tier_num+1])

            for client, indxs in train_client_to_indxs.items():
                train_indices_by_client[client_num] = indxs
                client_num += 1

        
        test_imgs_per_client = len(y_test) // (tiers * clients_per_tier)
        test_imgs_per_client_per_class = test_imgs_per_client // len(np.unique(y_test))
        print(test_imgs_per_client, test_imgs_per_client_per_class)

        for client_num in range(tiers * clients_per_tier):
            for class_num in test_indices_by_class.keys():
                selected_indxs = np.random.choice(test_indices_by_class[class_num], test_imgs_per_client_per_class, replace=False)
                remaining = list(set(test_indices_by_class[class_num]) - set(selected_indxs))
                test_indices_by_class[class_num] = remaining
                test_indices_by_client.setdefault(client_num, []).extend(selected_indxs)

        for client_num in train_indices_by_client.keys():
            train_indices_by_client[client_num] = np.array(train_indices_by_client[client_num])
            test_indices_by_client[client_num] = np.array(test_indices_by_client[client_num])

        return train_indices_by_client, test_indices_by_client


    def non_iid_prev_2(self, data_split=[1.0, 0.0], classes_per_client=5, uniform=False, repeats=0):
        """
        Generates Non-IID data. Returns test and train dictionaries of 
        client number to image index. Splits up the datasets by classes
        and allocates images of specific classes per client.
        """
        import random # FYI: Importing in functions is a bad idea

        total_train_imgs, total_test_imgs = 50000, 10000
        train_imgs_perclient, test_imgs_perclient = int(50000/self.num_users), int(10000/self.num_users)
        train_imgs_perclient *= (1 + repeats)
        num_shards = self.num_users * classes_per_client
        y_train, y_test = self.y_train.flatten(), self.y_test.flatten()

        # Key - Class Number (0-9), Value: index list of 
        # all images belonging to that class
        train_indices_by_class, train_common_indices_by_class = {}, {}
        test_indices_by_class = {}

        # Train dataset
        for img_index, class_num in enumerate(y_train):
            train_indices_by_class.setdefault(class_num, []).append(img_index)
            train_common_indices_by_class.setdefault(class_num, []).append(img_index)
            
        for img_index, class_num in enumerate(y_test):
            test_indices_by_class.setdefault(class_num, []).append(img_index)

        # Number of images per class per client
        num_test_imgs_perclass = int(total_test_imgs / num_shards)
        num_train_imgs_perclass = int(total_train_imgs / num_shards)

        # Key - client number, Value - List of image indices for that client
        train_idx_per_client = {}
        test_idx_per_client = {}

        classes_avail = list(self.classes)
        for client_num in range(self.num_users):

            # For every client, randomly choose a class and assign the images
            # of those classes to the client, until it it reaches max number of imgs
            # allowed for a single client
            train_idx_per_client[client_num] = []
            test_idx_per_client[client_num] = []
            classes_already_selected = []
            while len(train_idx_per_client[client_num]) < train_imgs_perclient:
                random_class = random.sample(classes_avail, k=1)[-1]

                if not uniform:                
                    train_indices = train_indices_by_class[random_class][:num_train_imgs_perclass]
                    train_indices_by_class[random_class] = train_indices_by_class[random_class][num_train_imgs_perclass:]

                    test_indices = test_indices_by_class[random_class][:num_test_imgs_perclass]
                    test_indices_by_class[random_class] = test_indices_by_class[random_class][num_test_imgs_perclass:]

                else :
                    if random_class in classes_already_selected:
                        if len(classes_avail) > 1:
                            continue
                    classes_already_selected.append(random_class)
                    train_indices = np.random.choice(train_indices_by_class[random_class], num_train_imgs_perclass, replace=False)
                    leftover_train_indices = list(set(train_indices_by_class[random_class]) - set(train_indices))
                    train_indices_by_class[random_class] = leftover_train_indices

                    test_indices = np.random.choice(test_indices_by_class[random_class], num_test_imgs_perclass, replace=False)
                    leftover_test_indices = list(set(test_indices_by_class[random_class]) - set(test_indices))
                    test_indices_by_class[random_class] = leftover_test_indices

                remaining = list(set(train_common_indices_by_class[random_class]) - set(train_indices))
                num_repeat_choices = min(len(remaining), num_train_imgs_perclass * repeats)
                train_indices = np.append(train_indices, np.random.choice(
                    remaining, num_repeat_choices, replace=False)).astype(int)
                train_idx_per_client[client_num].extend(train_indices)
                test_idx_per_client[client_num].extend(test_indices)

                if len(test_indices_by_class[random_class]) == 0:
                    classes_avail.remove(random_class)
        
        for client_num in range(self.num_users):
            train_idx_per_client[client_num] = np.array(train_idx_per_client[client_num])
            test_idx_per_client[client_num] = np.array(test_idx_per_client[client_num])

        self.log.debug("Generated Non-iid images."
                       " {} clients, {} classes per"
                       " client ->".format(self.num_users, classes_per_client))
        self.log.debug("{}/{} train/test images per client".format(
            num_train_imgs_perclass * classes_per_client, 
            num_test_imgs_perclass * classes_per_client
        ))
        
        return train_idx_per_client, test_idx_per_client
    
    
    def iid(self, data_split=[1.0, 0]):
        """
        Sample I.I.D. client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return: Two dictionaries (train/test) of iamge indices keyed by user num
        """
        
        self.log.debug("Generating IID indices for {} users...".format(self.num_users))
        
        train_items_per_user = int(len(self.x_train) / self.num_users)
        test_items_per_user  =  int(len(self.x_train) / self.num_users)
        dict_users_train, dict_users_test = {}, {}
        all_train_idxs, all_test_idxs = [i for i in range(len(self.x_train))], [i for i in range(len(self.x_test))]
        for i in range(self.num_users):
            dict_users_train[i] = np.random.choice(all_train_idxs, train_items_per_user, replace=False)
            all_idxs = list(set(all_train_idxs) - set(dict_users_train[i]))
            dict_users_test[i] = np.random.choice(all_test_idxs, test_items_per_user, replace=False)
            all_idxs = list(set(all_test_idxs) - set(dict_users_test[i]))

        self.log.debug("# of users: {}, train/test images per user: {}/{}".format(self.num_users,
                                                                train_items_per_user, test_items_per_user))
        self.log.info("IID Dataset generated")

        return dict_users_train, dict_users_test
    
    
    def non_iid_prev(self, data_split):
        """
        Sample non-I.I.D client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return: Two dictionaries (train/test) of iamge indices keyed by user num
        """
        self.log.debug("Generating Non-IID indices for {} users...".format(self.num_users))
        self.log.debug("Train/Test Ratio: {}, Shards: {}".format(data_split, self.num_shards))
        
        shards_per_user = int(self.num_shards / self.num_users)
        num_images_per_shard = int(len(self.y_train) / self.num_shards)
        idx_shard = [i for i in range(self.num_shards)]
        dict_users_train = {i: np.array([], dtype='int64') for i in range(self.num_users)}
        dict_users_test = {i: np.array([], dtype='int64') for i in range(self.num_users)}
        idxs = self.y_train.argsort()

        for i in range(self.num_users):
            rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            
            self.log.debug("Using shard/s {} for User {}".format(rand_set, i))
            
            for rand in rand_set:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], 
                     idxs[ 
                        rand * num_images_per_shard : 
                        rand * num_images_per_shard + int(num_images_per_shard * data_split[0])
                        ]
                    ), axis=0)
                dict_users_test[i] = np.concatenate(
                    (dict_users_test[i], 
                     idxs[
                        rand * num_images_per_shard + int(num_images_per_shard * data_split[0]) :
                        (rand + 1) * num_images_per_shard
                        ]
                    ), axis=0)
                
        # DEBUG LOGGING
        train_set_count = len(dict_users_train[0])
        test_set_count = len(dict_users_test[0])
        self.log.debug("Generated non-IID training/test dataset.")
        self.log.debug("Train/Test size: {}/{} per user, {} users".format(
                        train_set_count, test_set_count, self.num_users))
        self.log.info("Non-IID Dataset generated")

        return dict_users_train, dict_users_test


def get_classes_per_client_dict(clients, classes_per_client, uniform=True, classes=10, total_imgs=50000):
    """
    Returns an array (the size of num of clients). Every element
    of the array is a list of classes for the client at that index.
    e.g. [[0,1], [2,3], [4,5], ....] means client 1 will get classes 0,1,
    client 2 gets 2,3 and so on.
    """
    imgs_per_class = int(total_imgs / classes)
    imgs_per_client = int(total_imgs / clients)
    imgs_perclass_perclient = int(imgs_per_client / classes_per_client)
    num_avail_perclass = int(imgs_per_class / imgs_perclass_perclient)
    total_classes_avail = list(range(classes)) * num_avail_perclass
    classes_for_client = []
    for _ in range(clients):
        classes_selected = []
        while len(classes_selected) < classes_per_client:
            chosen = random.choice(total_classes_avail)
            if not uniform or chosen not in classes_selected: 
                classes_selected.append(chosen)
                total_classes_avail.remove(chosen)
        classes_for_client.append(classes_selected)

        # The code may end up with random selection where the last
        # index gets duplicates. In that case, return None
        if ( uniform
             and len(total_classes_avail) <= classes_per_client
             and len(total_classes_avail) != len(set(total_classes_avail))
           ): return None
    
    return classes_for_client

# All available Datasets
available_datasets = {
    "mnist"  : Mnist,
    "cifar10": Cifar10
}

# TODO: TESTING - REMOVE LATER
def test_MNIST_1():
    config = {"logging_level": "debug", "log_file": None, "num_users": 2, "num_shards": 2}
    dataset = Mnist(config)
    dataset.iid([0.8, 0.2])
    dataset.non_iid([0.8, 0.2])
    
def test_CIFAR10_1():
    config = {"logging_level": "debug", "log_file": None, "num_users": 2, "num_shards": 2}
    dataset = Cifar10(config)
    dataset.iid([0.9, 0.1])
    dataset.non_iid([0.2, 0.8]) 

def test_CIFAR10_2():
    config = {"logging_level": "debug", "log_file": None, "num_users": 100, "num_shards": 2}
    dataset = Cifar10(config)
    train, test = dataset.non_iid_subtiered(classes_per_client=10, uniform=True, repeat=0, tiers=5, subtiers=2)

    for num, indxs in train.items():
        print(num, len(indxs))

    for num, indxs in test.items():
        print(num, len(indxs))


if __name__ == "__main__":
    # test_MNIST_1()
    # test_CIFAR10_1()
    test_CIFAR10_2()
    # client_classes = get_classes_per_client_dict(clients=10, 
    #                                              classes_per_client=5, 
    #                                              uniform=True, classes=10, 
    #                                              total_imgs=10000)
    # print(client_classes)
