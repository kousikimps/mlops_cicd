
from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load
from typing import Dict, List, Optional, Sequence
#import yaml
#Project Directories

import regression_model

PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
#print(ROOT)
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASETS_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
#print(CONFIG_FILE_PATH)

class AppConfig(BaseModel):
    """
    Application level Config
    """
    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_name: str
    pipeline_save_file: str
class ModelConfig(BaseModel):
    """
    Model level configurations
    model training and feature engg
    """

    target: str
    variables_to_rename: Dict
    features: List[str]
    test_size: float
    random_state: int
    alpha: float
    categorical_vars_with_na_frequent: List[str]
    categorical_vars_with_na_missing: List[str]
    numerical_vars_with_na: List[str]
    temporal_vars: List[str]
    ref_var: str
    numericals_log_vars: Sequence[str]
    binarize_vars: Sequence[str]
    qual_vars: List[str]
    exposure_vars: List[str]
    finish_vars: List[str]
    garage_vars: List[str]
    categorical_vars: Sequence[str]
    qual_mappings: Dict[str, int]
    exposure_mappings: Dict[str, int]
    garage_mappings: Dict[str, int]
    finish_mappings: Dict[str, int]


class Config(BaseModel):
    app_config: AppConfig
    model_config: ModelConfig

def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        print("find_config_file-->",CONFIG_FILE_PATH)
        return CONFIG_FILE_PATH
    raise Exception(f"config file not found in the location {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path : Optional[Path] = None) -> YAML:
    if not cfg_path:
        cfg_path = find_config_file()
    print("cfg_path",cfg_path)
    if cfg_path:
        with open(cfg_path, 'r') as conf_file:
            #parsed_config = yaml.safe_load(conf_file)
            parsed_config = load(conf_file.read())
            #print(parsed_config)
            return parsed_config
    raise OSError(f"Config file not found in given location - {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) ->Config:
    """Run validation on config value"""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    #print("parsed config_file --- >", parsed_config)
    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )
    return _config

config = create_and_validate_config()
