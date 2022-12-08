import comet_ml
import sys


with open('experimentkey.txt') as f:
        first_line = f.readline()

EXPERIMENT_KEY = first_line

if (EXPERIMENT_KEY is not None):
        # There is one, but the experiment might not exist yet:
        api = comet_ml.API() # Assumes API key is set in config/env
        try:
            api_experiment = api.get_experiment_by_key(EXPERIMENT_KEY)
        except Exception:
            api_experiment = None

experiment = comet_ml.ExistingExperiment(
            previous_experiment=EXPERIMENT_KEY,
            log_env_details=True, # to continue env logging
            log_env_gpu=True,     # to continue GPU logging
            log_env_cpu=True,     # to continue CPU logging
        )
experiment.log_asset(sys.argv[1], step=None)
