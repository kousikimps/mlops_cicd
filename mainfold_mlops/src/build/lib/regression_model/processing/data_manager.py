import typing as t
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from regression_model import __version__ as _version
from regression_model.config.core import DATASETS_DIR, TRAINED_MODEL_DIR, config

def load_dataset(file_name: str)-> pd.DataFrame:
    df = pd.read_csv(Path(f"{DATASETS_DIR}/{file_name}"))
    df["MSSubClass"] = df["MSSubClass"].astype("O")
    # rename variables beginning with numbers to avoid syntax errors later
    transformed = df.rename(columns=config.model_config.variables_to_rename)
    #transformed = transformed.dropna()
    #print("printing_transform_data",transformed.shape)
    return transformed

def save_pipeline(*,pipeline_to_persists: Pipeline)-> None:
    print("save pipe line")
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR/save_file_name
    print(save_path)
    #remove_old_file(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persists,save_path)


def remove_old_file(*,files_to_keep: t.List[str]):
    donot_delete = files_to_keep+["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in donot_delete:
            model_file.unlink()


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


