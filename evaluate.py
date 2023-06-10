"""Evaluation script for measuring mean squared error."""

import subprocess
import sys
import json
import pathlib
import tarfile
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":

    print("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="./hf_model")

    print(os.listdir("./hf_model"))

    with open("./hf_model/evaluation.json") as f:
        eval_result = json.load(f)
    print(eval_result)
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(output_dir)
    evaluation_path = output_dir + "/evaluation.json"
    print(evaluation_path)
    with open(evaluation_path, "w") as f:
        print("writing json file")
        f.write(json.dumps(eval_result))
    print("file writted successfully")
