# fpu/examples/Cloud_Integration_Example.py

from fpu.src.cloud.Cloud_Integration_Services import AWSIntegration, GoogleCloudStorageIntegration, AzureBlobStorageIntegration

def aws_example():
    aws = AWSIntegration(access_key='your_access_key', secret_key='your_secret_key')
    file_name = 'example_file.txt'
    bucket_name = 'your_bucket_name'
    aws.upload_file(file_name, bucket_name)
    print(f"Uploaded {file_name} to AWS S3 bucket {bucket_name}")

def google_cloud_example():
    gcs = GoogleCloudStorageIntegration(project_name='your_project_name', credentials_path='path/to/credentials.json')
    file_name = 'example_file.txt'
    bucket_name = 'your_bucket_name'
    gcs.upload_blob(bucket_name, file_name, 'example_blob_name.txt')
    print(f"Uploaded {file_name} to Google Cloud Storage bucket {bucket_name}")

def azure_blob_example():
    azure = AzureBlobStorageIntegration(connection_string='your_connection_string')
    file_name = 'example_file.txt'
    container_name = 'your_container_name'
    azure.upload_file(container_name, file_name, 'azure_blob_name.txt')
    print(f"Uploaded {file_name} to Azure Blob Storage container {container_name}")

if __name__ == "__main__":
    aws_example()
    google_cloud_example()
    azure_blob_example()
