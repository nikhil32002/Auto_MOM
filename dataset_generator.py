import os
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
install('transformers')
install('datasets[s3]')
import datasets
import csv

class ResponseClassifier(datasets.GeneratorBasedBuilder):
    """AG News topic classification dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["AgreedToMeet", "BrochureSent", "ConnectLater", "Deactivated", "MovedOut", "OutOfOffice", "ReferredSomeone", "Retired", "SeekingMoreInfo", "SupportAutoResponse", "Unsubscribed"]), # here you have to pass the labels for your dataset. For now here 11 labels are provided, you have to change it depending your use case.
                }
            )
        )

    def _split_generators(self, dl_manager):
        import argparse
        import boto3

        parser = argparse.ArgumentParser()
        s3 = boto3.resource('s3')
        parser.add_argument("--s3_bucket", type=str, default="<bucket_name>")  # here you have to pass the bucket_name where you dataset is stored. 
        args, _ = parser.parse_known_args()
        s3_bucket = args.s3_bucket
        csv_file_path_train = 'train.csv'
        csv_file_path_test = 'test.csv'
        csv_s3_key_test = 'input/' + csv_file_path_test
        csv_s3_key_train = 'input/' + csv_file_path_train
        
        s3.Bucket(s3_bucket).download_file(Key=csv_s3_key_test, Filename=csv_file_path_test)
        s3.Bucket(s3_bucket).download_file(Key=csv_s3_key_train, Filename=csv_file_path_train)
        print('ls -ltr /opt/ml/processing/')
        os.system('ls -ltr /opt/ml/processing/')

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": csv_file_path_train}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": csv_file_path_test}),
        ]

    def _generate_examples(self, filepath):
      """Generate AG News examples."""
      with open(filepath, encoding="utf-8") as csv_file:

          csv_reader = csv.reader(
              csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
          for id_, row in enumerate(csv_reader):
              if id_ == 0:
                  continue
              index, label, text  = row
              yield id_ - 1, {"text": text, "label": label}