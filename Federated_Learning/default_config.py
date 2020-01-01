NUM_TIERS = 1
CLIENTS_PER_TIER = 100
CLIENTS_TRAINED_PER_ROUND = 100
# Selection probability Distribution for the corresponding Tiers
TIER_DISTRIBUTION = [0, 0, 0, 0, 1]
# Generated from the other configurations
TOTAL_CLIENTS = NUM_TIERS * CLIENTS_PER_TIER
NUM_SHARDS = 2 * TOTAL_CLIENTS

config = {
  "server": {
    "model": "cifar10_cnn",
    "host": "0.0.0.0",
    "port": 6002,
    "sync": False,
    "policy": False,
    "tier_distribution": TIER_DISTRIBUTION,
    "clients_trained_per_round": CLIENTS_TRAINED_PER_ROUND,
    "total_rounds": 20,
    "eval_every_n_rounds": 1,
    "async_bound": 0,
    "training_batch_size": 8,
    "logging_level": "debug",
    "log_file": None
  },
  "clients": {
    "server_host": "128.143.67.172",
    "server_port": 6002,
    "tiers": NUM_TIERS,
    "clients_per_tier": CLIENTS_PER_TIER,
    "logging_level": "debug",
    "log_file": None
  },
  "dataset": {
    "name": "cifar10",
    "iid": False,
    "subtiered": False,          # Only for Non-IID, Data heterogenity
    "uniform": True,            # Only for Non-IID
    "classes_per_client": 5,    # Only for Non-IID
    "repeats": 0,               # Only for Non-IID
    "num_users": TOTAL_CLIENTS,
    "num_shards": NUM_SHARDS,
    "data_split": [0.9, 0.1],   # Does not matter for Cifar10
    "logging_level": "debug",
    "log_file": None
  }
}
