# fpu/src/cloud/Cloud_Integration_Services.py

import boto3
import google.cloud.storage as gcs
import azure.storage.blob as azureblob
import json

class AWSIntegration:
    def __init__(self, access_key, secret_key, region='us-east-1'):
        self.s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)

    def upload_file(self, file_name, bucket_name, object_name=None):
        if object_name is None:
            object_name = file_name
        response = self.s3.upload_file(file_name, bucket_name, object_name)
        return response

    def download_file(self, bucket_name, object_name, file_name):
        self.s3.download_file(bucket_name, object_name, file_name)

class GoogleCloudStorageIntegration:
    def __init__(self, project_name, credentials_path):
        self.project_name = project_name
        self.credentials = json.load(open(credentials_path))
        self.client = gcs.Client(project=project_name, credentials=self.credentials)

    def upload_blob(self, bucket_name, source_file_name, destination_blob_name):
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

    def download_blob(self, bucket_name, source_blob_name, destination_file_name):
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

class AzureBlobStorageIntegration:
    def __init__(self, connection_string):
        self.blob_service_client = azureblob.BlobServiceClient.from_connection_string(connection_string)

    def upload_file(self, container_name, file_name, blob_name):
        blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        with open(file_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    def download_file(self, container_name, blob_name, file_path):
        blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        with open(file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

# Example usage
if __name__ == "__main__":
    # AWS S3 Example
    aws = AWSIntegration(access_key='your_access_key', secret_key='your_secret_key')
    aws.upload_file('local_file.txt', 'your_bucket_name', 's3_object_name.txt')

    # Google Cloud Storage Example
    gcs = GoogleCloudStorageIntegration(project_name='your_project_name', credentials_path='path/to/credentials.json')
    gcs.upload_blob('your_bucket_name', 'local_file.txt', 'gcs_object_name.txt')

    # Azure Blob Storage Example
    azure = AzureBlobStorageIntegration(connection_string='your_connection_string')
    azure.upload_file('your_container_name', 'local_file.txt', 'azure_blob_name.txt')
