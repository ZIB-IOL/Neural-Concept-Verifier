import argparse

import config.config_dataclass as cfg_dataclass
from config import argparser # type: ignore


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merlin-Arthur Training")
    argparser.add_trainer_args(parser) # type: ignore
    argparser.add_dataset_args(parser) # type: ignore
    argparser.add_boolean_args(parser) # type: ignore
    argparser.add_model_args(parser) # type: ignore
    argparser.add_feature_selector_args(parser) # type: ignore

    return parser.parse_args()


def create_config_instances(args):
    """Create instances of the configuration classes and add them to a dictionary."""
    config_dict = {}
    # Create an instance of each configuration class and add it to the dictionary.
    config_dict["trainer_config"] = cfg_dataclass.TrainerConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.TrainerConfig.__annotations__}
    )
    config_dict["dataset_config"] = cfg_dataclass.DatasetConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.DatasetConfig.__annotations__}
    )
    config_dict["boolean_config"] = cfg_dataclass.BooleanConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.BooleanConfig.__annotations__}
    )
    config_dict["model_config"] = cfg_dataclass.ModelConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.ModelConfig.__annotations__}
    )
    config_dict["feature_selector_config"] = cfg_dataclass.FeatureSelectorConfig(
        **{k: getattr(args, k) for k in cfg_dataclass.FeatureSelectorConfig.__annotations__}
    )

    return config_dict


def parse_arguments_and_create_config():
    args = parse_arguments()
    return create_config_instances(args)