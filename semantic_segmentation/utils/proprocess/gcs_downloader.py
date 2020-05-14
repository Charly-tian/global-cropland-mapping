
import os
from tqdm import tqdm
from google.cloud import storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    "D:/Cloud/google_application_credentials/gee-segmentation-4ecb572c566f.json"


def download_folder_blobs(bucket_name, folder_name, dst_folder):
    """
    download all files in a folder from google cloud storage to local machine

    E.g.:
    ```
        download_blobs(bucket_name='seg-samples', folder_name='LC08_100', dst_folder='D:/test')
    ```
    # Args:
        project_name(string): The name of project.
        bucket_name(string): The name of bucket.
        folder_name(string): The name of folder.
        dst_folder(string): The destination path in local machine.
    # Returns:
        None.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()

    pbar = tqdm(blobs)
    for blob in pbar:
        public_url = blob.public_url
        if public_url.startswith('https://storage.googleapis.com/{}/{}'.format(bucket_name, folder_name)):
            file_basename = os.path.basename(public_url)
            if len(file_basename) == 0:
                continue
            blob_src_name = '{}/{}'.format(folder_name, file_basename)
            blob_dst_name = '{}/{}'.format(dst_folder, file_basename)
            if os.path.exists(blob_dst_name):
                pbar.set_description('`{}` already exists.'.format(blob_dst_name))
                continue
            blob = bucket.blob(blob_src_name)
            blob.download_to_filename(blob_dst_name)
            pbar.set_description('downloaded `{}` to `{}`'.format(blob_src_name, blob_dst_name))
