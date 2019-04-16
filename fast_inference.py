import argparse
import json
import os
import six

import tensorflow as tf

from google.protobuf import text_format

from opennmt import __version__
from opennmt.models import catalog
from opennmt.runner import Runner
from opennmt.config import load_model
from opennmt.utils.misc import classes_in_module

import yaml

def load_config(config_path, config=None):
  """Loads configuration files.
  Args:
    config_paths: A list of configuration files.
    config: A (possibly non empty) config dictionary to fill.
  Returns:
    The configuration dictionary.
  """
  return yaml.load(open(config_path, 'rb').read())


def get_runners():
    config_one = load_config('avg_data.yml')

    model_one = load_model(
      config_one["model_dir"],
      model_file='',
      model_name='',
      serialize_model=True)


    session_config_one = tf.ConfigProto(
      intra_op_parallelism_threads=0,
      inter_op_parallelism_threads=0,
      gpu_options=tf.GPUOptions(
          allow_growth=False))

    runner_one = Runner(
      model_one,
      config_one,
      seed=None,
      num_devices=1,
      session_config=session_config_one,
      auto_config=True)


    config_two = load_config('avg_data_rev.yml')

    model_two = load_model(
      config_two["model_dir"],
      model_file='',
      model_name='',
      serialize_model=True)


    session_config_two = tf.ConfigProto(
      intra_op_parallelism_threads=0,
      inter_op_parallelism_threads=0,
      gpu_options=tf.GPUOptions(
          allow_growth=False))

    runner_two = Runner(
      model_two,
      config_two,
      seed=None,
      num_devices=1,
      session_config=session_config_two,
      auto_config=True)

    return runner_one, runner_two


if __name__=='__main__':
    runner_one.infer(
    features_file = 'maori_input.txt',
    predictions_file='english_new.txt',
    checkpoint_path='run/avg'
    )


    runner_two.infer(
    features_file = 'english.txt',
    predictions_file='maori_new.txt',
    checkpoint_path='run_rev/avg'
    )
