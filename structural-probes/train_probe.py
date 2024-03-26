from run_experiment import execute_experiment
from argparse import ArgumentParser
import yaml

argp = ArgumentParser()
argp.add_argument('experiment_config')
cli_args = argp.parse_args()
yaml_args= yaml.load(open(cli_args.experiment_config), Loader=yaml.SafeLoader)
execute_experiment(yaml_args, True, False)