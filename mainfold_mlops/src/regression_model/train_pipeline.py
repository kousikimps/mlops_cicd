#!/usr/bin/env python
import numpy as np
from regression_model.config.core import config
from sklearn.model_selection import train_test_split
from regression_model.processing.data_manager import load_dataset,save_pipeline
from regression_model.pipeline import price_pipe

def run_training():
    data = load_dataset(file_name=config.app_config.training_data_file)
    #devide train test
    X_train,X_test,y_train,y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state
    )
    y_train = np.log(y_train)
    price_pipe.fit(X_train,y_train)
    print("calling save pipe line")
    print(price_pipe)
    save_pipeline(pipeline_to_persists=price_pipe)

if __name__ == "__main__":
    run_training()