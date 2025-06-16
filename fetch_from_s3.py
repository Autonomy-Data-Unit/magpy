import boto3
from pathlib import Path
from magpy import const
import tempfile
import tarfile
import shutil
import os

bucket_name = 'adu-project-data'
s3_path = 'magpy/store_compressed.tar.gz'

s3 = boto3.client('s3')

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert bytes to MB

if os.path.exists(const.store_path):
    folder_size = get_folder_size(const.store_path)
    user_input = input(f"The directory {const.store_path} already exists and is {folder_size:.2f} MB. Do you really want to fetch and overwrite it? (yes/no): ")
    if user_input.lower() != 'yes':
        print("Fetch operation aborted.")
        exit()

# Get the ETag of the uploaded file and save it
s3 = boto3.client('s3')
response = s3.head_object(Bucket=bucket_name, Key=s3_path)
remote_etag = response['ETag'].strip('"')  # Remove the quotes from the ETag
local_etag_path = Path('store_compressed_etag.txt')
local_etag = local_etag_path.read_text() if local_etag_path.exists() else None
if remote_etag == local_etag:
    print("The directory is already up to date.")
    exit()

# Create a temporary file to store the downloaded tar.gz archive
with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as temp_file:
    temp_file_path = temp_file.name

    # Download the tar.gz file from S3
    s3.download_file(bucket_name, s3_path, temp_file_path)

    # Remove the existing store directory
    if os.path.exists(const.store_path):
        shutil.rmtree(const.store_path)

    # Extract the tar.gz archive to const.store_path
    with tarfile.open(temp_file_path, "r:gz") as tar:
        tar.extractall(path=const.root_path)

# Save the ETag of the downloaded file
with open('store_compressed_etag.txt', 'w') as f: f.write(remote_etag)