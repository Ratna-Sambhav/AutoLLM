import argparse
import boto3
import os

def upload_files_to_s3(local_dir, bucket_name, s3_dir, access_key_id, secret_access_key):
    # Create an S3 client with appropriate credentials
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key
    )

    # Iterate over files in the local directory and upload them to S3
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            # Construct the local file path
            local_file_path = os.path.join(root, filename)

            # Construct the corresponding S3 object key preserving directory structure
            s3_object_key = os.path.relpath(local_file_path, local_dir)
            s3_object_key = os.path.join(s3_dir, s3_object_key)

            # Upload the file to S3
            try:
                s3_client.upload_file(local_file_path, bucket_name, s3_object_key)
                print(f"Uploaded '{local_file_path}' to '{s3_object_key}' in bucket '{bucket_name}'")
            except Exception as e:
                print(f"Failed to upload '{local_file_path}' to '{s3_object_key}' in bucket '{bucket_name}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload files to an S3 bucket while maintaining directory structure.")
    parser.add_argument("local_dir", help="Path to the local directory containing files to upload.")
    parser.add_argument("bucket_name", help="Name of the S3 bucket.")
    parser.add_argument("s3_dir", help="Name of the directory to create in the S3 bucket.")
    parser.add_argument("--access_key_id", help="AWS access key ID.")
    parser.add_argument("--secret_access_key", help="AWS secret access key.")
    args = parser.parse_args()

    upload_files_to_s3(args.local_dir, args.bucket_name, args.s3_dir, args.access_key_id, args.secret_access_key)
