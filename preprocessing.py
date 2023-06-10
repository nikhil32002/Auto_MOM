import os
import pandas as pd 
import subprocess
import sys
import boto3
import argparse
import dataset_generator_package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__=='__main__':
    
    install('torch')
    install('transformers==4.11')
    install('datasets[s3]==1.12.1')
    
    base_dir = "/opt/ml/processing"
    tokenizer_name = "distilbert-base-uncased"
    print(tokenizer_name)
    
    parser = argparse.ArgumentParser()
    s3 = boto3.resource('s3')
    parser.add_argument("--s3_bucket", type=str, default="<bucket_name>") # here you have to pass the bucket_name where you dataset is stored.
    args, _ = parser.parse_known_args()
    s3_bucket = args.s3_bucket
    csv_file_path_train = 'train.csv'
    csv_file_path_test = 'test.csv'
    csv_s3_key_test = 'input/' + csv_file_path_test
    csv_s3_key_train = 'input/' + csv_file_path_train

    df = pd.read_csv(
        f"{base_dir}/input/<dataset_file_name>" # here you have to give the name of dataset file eg., complete_dataset.csv
    )

    from sklearn.model_selection import train_test_split
    train,test = train_test_split(df, test_size=0.20,random_state=42)
    train.to_csv(csv_file_path_train,index=False)
    test.to_csv(csv_file_path_test,index=False)
    s3.Bucket(s3_bucket).upload_file(Filename=csv_file_path_test, Key=csv_s3_key_test)
    s3.Bucket(s3_bucket).upload_file(Filename=csv_file_path_train, Key=csv_s3_key_train)

    from transformers import AutoTokenizer
    from datasets import load_dataset, load_from_disk
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_dataset("/opt/ml/processing/input/code/dataset_generator_package/dataset_generator.py")
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=396)

    encoded_dataset = dataset.map(preprocess_function, batched=True, batch_size=7000)
    print(encoded_dataset["test"])
    print(encoded_dataset["train"])
    test_dataset=encoded_dataset["test"]
    print(test_dataset.shape)
    
    train_dataset=encoded_dataset["train"]
    print(train_dataset.shape)
    
    train_dataset.save_to_disk('/opt/ml/processing/train')
    test_dataset.save_to_disk('/opt/ml/processing/test')
    
    print('ls -ltr /opt/ml/processing/train/')
    os.system('ls -ltr /opt/ml/processing/train/')
    
    print('ls -ltr /opt/ml/processing/test/')
    os.system('ls -ltr /opt/ml/processing/test/')
 
    print(len(train_dataset))
    print(len(test_dataset))
    
    print('loading')
    train_dataset1 = load_from_disk('/opt/ml/processing/train/')
    test_dataset1 = load_from_disk('/opt/ml/processing/test/')

    print(len(train_dataset1))
    print(len(test_dataset1))