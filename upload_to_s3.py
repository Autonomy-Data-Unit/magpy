from pathlib import Path
import boto3
from magpy import const
import tarfile
import tempfile

bucket_name = 'adu-project-data'
s3_path = 'magpy/store_compressed.tar.gz'

# Create a temporary file to store the tar.gz archive
with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as temp_file:
    temp_file_path = temp_file.name

    print("Compressing the store directory...")
    # Create a tar.gz archive of the folder const.store
    with tarfile.open(temp_file_path, "w:gz") as tar:
        tar.add(const.store_path, arcname=Path(const.store_path).name)

    print("Uploading the compressed store directory to S3...")
    s3 = boto3.client('s3')
    s3.upload_file(temp_file_path, bucket_name, s3_path)
    
    # Get the ETag of the uploaded file and save it to .tmp
    s3 = boto3.client('s3')
    response = s3.head_object(Bucket=bucket_name, Key=s3_path)
    etag = response['ETag'].strip('"')  # Remove the quotes from the ETag
    with open('store_compressed_etag.txt', 'w') as f: f.write(etag)