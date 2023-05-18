from typing import Optional, List, Tuple, Dict

import os
import sys
import random
import threading
import logging
import shutil

import boto3
import botocore
from botocore.exceptions import ClientError

from pepper.utils import create_if_not_exist


def sample_images_v1(source_dir: str, target_dir: str, n_samples: int):
    """DEPRECATED use `sample_images` instead
    Samples a specified number of images from each folder in the source
    directory and copies them to the target directory.

    Parameters
    ----------
    source_dir : str
        The source directory containing subfolders with images.
    target_dir : str
        The target directory to copy the sampled images to.
    n_samples : int
        The number of images to sample from each folder.

    """

    # Get the list of subdirectories
    subdirs = [
        subdir for subdir in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, subdir))
    ]

    # Determine the distribution of the number of images per folder
    n_images_per_folder, remainder = divmod(n_samples, len(subdirs))

    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)

        # Get the list of images in the folder
        images = [
            image for image in os.listdir(subdir_path)
            if os.path.isfile(os.path.join(subdir_path, image))
        ]

        # Select images randomly
        n_images = n_images_per_folder + (remainder > 0)
        if len(images) <= n_images:
            selected_images = images
        else:
            selected_images = random.sample(images, n_images)

        # If an extra image was selected in this folder,
        # decrement the remainder by 1
        if n_images == len(selected_images):
            if remainder > 0:
                remainder -= 1
        # If the folder was empty or had insufficient available images,
        # increase the remainder by the unselected quantity
        else:
            remainder += n_images - len(selected_images)

        if len(selected_images) > 0:
            create_if_not_exist(os.path.join(target_dir, subdir))

        # Copy the selected images to the target directory
        for image in selected_images:
            source_path = os.path.join(subdir_path, image)
            target_path = os.path.join(target_dir, subdir, image)
            shutil.copyfile(source_path, target_path)


def list_subdirs(root_path: str) -> List[str]:
    """Returns a list of subdirectories in the specified root path.

    Parameters
    ----------
    root_path : str
        The root path to scan for subdirectories.

    Returns
    -------
    List[str]
        A list of subdirectory names.
    """
    return [
        subdir for subdir in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, subdir))
    ]


def list_s3_subdirs(bucket_name: str, root_path: str = "") -> List[str]:
    """Returns a list of subdirectories in the specified S3 bucket and root path.

    Parameters
    ----------
    bucket_name : str
        The name of the S3 bucket.
    root_path : str
        The root path in the S3 bucket to scan for subdirectories.
        Defaults to ''.

    Returns
    -------
    List[str]
        A list of subdirectory names.
    """
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=root_path,
        Delimiter='/'
    )
    return [
        obj["Prefix"].split("/")[-2]
        for obj in response.get("CommonPrefixes", [])
    ]


def count_files(
    root_path: str,
    subdirs: Optional[List[str]] = None
) -> Tuple[Dict[str, int], int]:
    """Counts the number of files in each subdirectory of the specified root
    path.

    Parameters
    ----------
    root_path : str
        The root path to scan for subdirectories.
    subdirs : List[str], optional
        List of subdirectories. If provided, the function will use this list
        instead of retrieving it from the root path.

    Returns
    -------
    Tuple[Dict[str, int], int]
        A tuple containing a dictionary of subdirectory names and their
        corresponding file counts, and the total number of files.
    """
    if subdirs is None:
        subdirs = list_subdirs(root_path)

    file_counts = {}
    total_files = 0
    for subdir in subdirs:
        subdir_path = os.path.join(root_path, subdir)
        files = [
            file for file in os.listdir(subdir_path)
            if os.path.isfile(os.path.join(subdir_path, file))
        ]
        file_counts[subdir] = len(files)
        total_files += len(files)

    return file_counts, total_files



# See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/list_objects_v2.html
# NB : A more generic approach would require implementing a recursive algorithm.
# Note: The S3 API does not support XPath-like syntax for deep exploration of bucket structures.
# Therefore, it is necessary to make successive requests per level.
# In other words, it is the developer's responsibility to construct their own tree traversal logic.
def count_s3_objects(
    bucket_name: str,
    root_path: str = "",
    subdirs: Optional[List[str]] = None
) -> Tuple[Dict[str, int], int]:
    """Counts the number of objects in each subdirectory
    of the specified S3 bucket and root path.

    Parameters
    ----------
    bucket_name : str
        The name of the S3 bucket.
    root_path : str, optional
        The root path in the S3 bucket to scan for subdirectories.
        Defaults to ''.
    subdirs : List[str], optional
        List of subdirectories. If provided, the function will use this list
        instead of retrieving it from the S3 bucket.

    Returns
    -------
    Tuple[Dict[str, int], int]
        A tuple containing a dictionary of subdirectory names and their
        corresponding object counts, and the total number of objects.
    """
    if subdirs is None:
        subdirs = list_s3_subdirs(bucket_name, root_path)

    s3_client = boto3.client("s3")
    object_counts = {}
    total_objects = 0

    for subdir in subdirs:
        subdir_response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=subdir
        )
        subdir_objects = subdir_response.get("Contents", [])
        # print(subdir, subdir_objects)
        object_counts[subdir] = len(subdir_objects)
        total_objects += len(subdir_objects)

    return object_counts, total_objects



def compute_target_dist(
    subdirs: List[str],
    image_counts: Dict[str, int],
    n_total: int,
    n_samples: int
) -> Dict[str, int]:
    """Adjusts the number of images to sample in each folder based on the
    remaining samples.

    Parameters
    ----------
    subdirs : List[str]
        List of subdirectory names.
    image_counts : Dict[str, int]
        Dictionary mapping subdirectory names to the total number of images in
        each folder.
    n_total : int
        The total number of images.
    n_samples : int
        The number of images to sample from each folder.

    Returns
    -------
    target_dist : Dict[str, int]
        Dictionary mapping subdirectory names to the number of images to sample
        from each folder.
    """
    
    # Calculate the number of images to sample from each folder
    target_dist = {}  # The target distribution
    remaining_samples = n_samples  # The remaining number of samples to adjust
    for subdir in subdirs:
        n_images = round((n_samples / n_total) * image_counts[subdir])
        target_dist[subdir] = n_images
        remaining_samples -= n_images
        
    if remaining_samples == 0:
        return target_dist

    # Determine the adjustment direction and condition
    adjustment = 1 if remaining_samples > 0 else -1
    is_adjustable = (
        (lambda subdir: target_dist[subdir] < image_counts[subdir])
        if remaining_samples > 0 else
        (lambda subdir: target_dist[subdir] > 0)
    )

    # Continue adjusting samples until remaining_samples reaches 0
    while remaining_samples != 0:
        # Select folders to adjust
        folders_to_adjust = [
            subdir for subdir in subdirs
            if is_adjustable(subdir)
        ]
        folders_to_adjust = random.sample(
            folders_to_adjust,
            min(abs(remaining_samples), len(folders_to_adjust))
        )

        # Adjust the number of images in the selected folders
        for subdir in folders_to_adjust:
            target_dist[subdir] += adjustment
            remaining_samples -= adjustment

    return target_dist


def copy_files(
    root_path: str,
    target_path: str,
    subdirs: List[str],
    target_dist: Dict[str, int]
) -> None:
    """Copies selected files from the source directory to the target directory
    based on the specified distribution.

    Parameters
    ----------
    root_path : str
        The root path of the source directory.
    target_path : str
        The target directory to copy the selected files to.
    subdirs : List[str]
        List of subdirectory names.
    target_dist : Dict[str, int]
        Dictionary mapping subdirectory names to the number of files to copy
        from each folder.

    Returns
    -------
    None

    """
    for subdir in subdirs:
        subdir_path = os.path.join(root_path, subdir)
        selected_files = random.sample(
            os.listdir(subdir_path),
            target_dist[subdir]
        )

        # Create the subdirectory in the target directory if it doesn't exist
        if len(selected_files) > 0:
            create_if_not_exist(os.path.join(target_path, subdir))

        # Copy the selected files to the target directory
        for file in selected_files:
            source_path = os.path.join(subdir_path, file)
            target_file_path = os.path.join(target_path, subdir, file)
            shutil.copyfile(source_path, target_file_path)


def copy_files_to_s3(
    root_path: str,
    target_bucket: str,
    subdirs: List[str],
    target_dist: Dict[str, int]
) -> None:
    """Copies selected files from the source directory to Amazon S3 based on
    the specified distribution.

    Parameters
    ----------
    root_path : str
        The root path of the source directory.
    target_bucket : str
        The name of the target S3 bucket.
    subdirs : List[str]
        List of subdirectory names.
    target_dist : Dict[str, int]
        Dictionary mapping subdirectory names to the number of files to copy
        from each folder.

    Returns
    -------
    None

    """
    s3_client = boto3.client("s3")

    for subdir in subdirs:
        subdir_path = os.path.join(root_path, subdir)
        selected_files = random.sample(
            os.listdir(subdir_path),
            target_dist[subdir]
        )

        # Copy the selected files to Amazon S3
        for file in selected_files:
            source_path = os.path.join(subdir_path, file)
            target_key = f"{subdir}/{file}"
            s3_client.upload_file(source_path, target_bucket, target_key)


def sample_images(
    root_path: str,
    target_path: str,
    n_samples: int
):
    """Samples a specified number of images from each folder in the source
    directory and copies them to the target directory.

    Parameters
    ----------
    source_path : str
        The source directory containing subfolders with images.
    target_path : str
        The target directory to copy the sampled images to.
    n_samples : int
        The number of images to sample from each folder.

    Raises
    ------
    ValueError
        If n_samples is greater than the total number of images available.

    Returns
    -------
    target_dist : Dict[str, int]
        Dictionary mapping subdirectory names to the number of images sampled
        from each folder.
    """

    # Get the list of subdirectories
    subdirs = list_subdirs(root_path)

    # Count the number of images in each folder
    image_counts, n_total = count_files(root_path, subdirs)

    # Check if n_samples is greater than the total number of images
    if n_samples > n_total:
        raise ValueError(
            "n_samples is greater than the total number of images available."
        )

    # Calculate the number of images to sample from each folder
    target_dist = compute_target_dist(subdirs, image_counts, n_total, n_samples)

    # Copy selected files from the source directory to the target directory
    copy_files(root_path, target_path, subdirs, target_dist)
    
    return target_dist


def sample_images_local_to_s3(
    root_path: str,
    bucket_name: str,
    #target_path: str,
    n_samples: int
):
    """Samples a specified number of images from each folder in the source
    directory and copies them to the target S3 directory.

    Parameters
    ----------
    source_path : str
        The source directory containing subfolders with images.
    target_path : str
        The target directory to copy the sampled images to.
    n_samples : int
        The number of images to sample from each folder.

    Raises
    ------
    ValueError
        If n_samples is greater than the total number of images available.

    Returns
    -------
    target_dist : Dict[str, int]
        Dictionary mapping subdirectory names to the number of images sampled
        from each folder.
    """

    # Get the list of subdirectories
    subdirs = list_subdirs(root_path)

    # Count the number of images in each folder
    image_counts, n_total = count_files(root_path, subdirs)

    # Check if n_samples is greater than the total number of images
    if n_samples > n_total:
        raise ValueError(
            "n_samples is greater than the total number of images available."
        )

    # Calculate the number of images to sample from each folder
    target_dist = compute_target_dist(subdirs, image_counts, n_total, n_samples)

    # Copy selected files from the source directory to the target directory
    #copy_files(root_path, target_path, subdirs, target_dist)
    copy_files_to_s3(root_path, bucket_name, subdirs, target_dist)
    
    return target_dist


def clean_directory(root_path: str) -> None:
    """Cleans the specified directory by deleting all files and subdirectories.

    Parameters
    ----------
    root_path : str
        The root directory to clean.

    Returns
    -------
    None

    """
    image_counts, n_files = count_files(root_path)
    n_dirs = len(image_counts)
    print(
        f"The directory {root_path}\n"
        f"contains {n_files} files and {n_dirs} directories."
    )

    first_confirm = input(
        f"Warning: This operation will delete all files and subdirectories in {root_path}. "
        "Are you sure you want to continue? (Y/N): "
    )
    if first_confirm.lower() != "y":
        print("Operation canceled.")
        return

    second_confirm = input(
        "Are you absolutely sure you want to delete "
        f"all these  {n_files} files and {n_dirs} subdirectories? "
        "This action is irreversible. "
        f"Type 'confirm deletion of {n_files} files' to proceed: "
    )
    if second_confirm.lower() != f"confirm deletion of {n_files}":
        print("Operation canceled.")
        return

    if os.path.exists(root_path):
        shutil.rmtree(root_path)
        print(f"The directory {root_path} has been successfully deleted.")
    else:
        print(f"The directory {root_path} does not exist.")



def clean_s3_directory(bucket_name: str, root_path: str) -> None:
    """Cleans the specified directory on an S3 bucket by deleting all files and
    subdirectories.

    Parameters
    ----------
    bucket_name : str
        The name of the S3 bucket.
    root_path : str
        The root directory in the S3 bucket to clean.

    Returns
    -------
    None

    """
    subdirs = list_s3_subdirs(bucket_name, root_path)
    image_counts, n_files = count_s3_objects(bucket_name, root_path, subdirs)
    n_dirs = len(image_counts)
    print(
        f"The S3 directory s3://{bucket_name}/{root_path}\n"
        f"contains {n_files} files and {n_dirs} directories."
    )

    first_confirm = input(
        "Warning: This operation will delete all files "
        f"and subdirectories in s3://{bucket_name}/{root_path}. "
        "Are you sure you want to continue? (Y/N): "
    )
    if first_confirm.lower() != "y":
        print("Operation canceled.")
        return

    second_confirm = input(
        f"Are you absolutely sure you want to delete "
        f"all these {n_files} files and {n_dirs} subdirectories? "
        f"This action is irreversible. "
        f"Type 'confirm deletion of {n_files} files' to proceed: "
    )
    if second_confirm.lower() != f"confirm deletion of {n_files} files":
        print("Operation canceled.")
        return

    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=root_path)
    images = response.get("Contents", [])
    for image in images:
        s3_client.delete_object(Bucket=bucket_name, Key=image["Key"])

    print(f"The S3 directory s3://{bucket_name}/{root_path} has been successfully deleted.")



"""V3 avec S3 intégré
"""

def is_s3_path(path: str) -> bool:
    """Checks if a path is an AWS S3 path.

    Parameters
    ----------
    path : str
        The path to check.

    Returns
    -------
    bool
        True if the path is an AWS S3 path, False otherwise.
    """
    return path.startswith('s3://')


def read_file(path: str) -> str:
    """Reads the contents of a file.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    str
        The contents of the file.
    """
    if is_s3_path(path):
        s3 = boto3.resource("s3")
        bucket, key = path[5:].split("/", 1)
        try:
            obj = s3.Object(bucket, key)
            return obj.get()["Body"].read().decode("utf-8")
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Failed to read file from S3: {e}") from e
    else:
        with open(path, 'r') as file:
            return file.read()


def write_file(path: str, contents: str):
    """Writes contents to a file.

    Parameters
    ----------
    path : str
        The path to the file.
    contents : str
        The contents to write to the file.
    """
    if is_s3_path(path):
        s3 = boto3.resource("s3")
        bucket, key = path[5:].split("/", 1)
        try:
            obj = s3.Object(bucket, key)
            obj.put(Body=contents.encode("utf-8"))
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Failed to write file to S3: {e}") from e
    else:
        with open(path, "w") as file:
            file.write(contents)



def get_s3_object_size(bucket_name: str, object_name: str):
    s3_client = boto3.client("s3")
    try:
        response = s3_client.head_object(
            Bucket=bucket_name,
            Key=object_name
        )
        return response["ContentLength"]
    except Exception as e:
        print(f"Failed to retrieve object size: {e}")
        return None



# TODO: mais pas dans le contexte de pression que me mets OC
# compléter pour l'adapter à un upload ou download massif
# utiliser tqdm :
# init: self._pbar = tqdm(total=self._size, unit="B", unit_scale=True)
# call: self._pbar.update(bytes_amount)
class ProgressPercentage(object):

    def __init__(
        self,
        root_path: str,
        image_class: str,
        image_id: str,
        image_ext: str,
        bucket_name: str,
        download: bool
    ):
        """Initializes ProgressPercentage object.

        Parameters
        ----------
        root_path : str
            Root path of the file.
        image_class : str
            Image class.
        image_id : str
            Image ID.
        image_ext : str
            Image file extension.
        bucket_name : str
            S3 bucket name.
        download : bool
            Flag indicating whether it's a download or upload.
        """
        self._root_path = root_path
        self._image_class = image_class
        self._image_id = image_id
        self._image_ext = image_ext
        self._bucket_name = bucket_name
        self._download = download
        self._image_basename = f"{self._image_id}.{self._image_ext}"
        self._image_filepath = os.path.join(
            root_path, image_class, self._image_basename
        )
        self._image_name = f"{self._image_class}/{self._image_id}"
        if self._download:
            self._size = get_s3_object_size(self._bucket_name, self._image_name)
        else:
            self._size = float(os.path.getsize(self._image_filepath))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount: int):
        """Updates the progress based on the bytes amount.

        Parameters
        ----------
        bytes_amount : int
            Number of bytes processed.
        """
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                f"\r{self._image_name}  {self._seen_so_far} / {self._size}  "
                f"({percentage:.2f}%)"
            )
            sys.stdout.flush()




def upload_image_object(
    root_path: str,
    image_class: str,
    image_id: str,
    image_ext: Optional[str] = "jpg",
    bucket_name: Optional[str] = "pepper-labs-fruits",
    image_object_name: Optional[str] = None,
    Callback: Optional[ProgressPercentage] = None
):
    """Uploads an image file to an S3 bucket.

    Parameters
    ----------
    root_path : str
        Root path where the image file is located.
    image_class : str
        Class name of the image.
    image_id : str
        Identifier of the image.
    image_ext : str, optional
        Extension of the image file. Default is "jpg".
    bucket_name : str, optional
        Name of the S3 bucket. Default is "pepper-labs-fruits".
    image_object_name : str, optional
        S3 object name. If not specified, the default is
        "{image_class}/{image_id}.{image_ext}".
    Callback : ProgressPercentage, optional
        Callback function to track the upload progress. If not specified,
        a ProgressPercentage object is created.

    Returns
    -------
    bool
        True if the file was uploaded successfully, False otherwise.
    """
    image_basename = f"{image_id}.{image_ext}"

    # If S3 object_name was not specified, use the default
    if image_object_name is None:
        image_object_name = f"{image_class}/{image_basename}"

    image_filepath = os.path.join(root_path, image_class, image_basename)

    # If S3 object_name was not specified, use the default
    if image_object_name is None:
        image_object_name = f"{image_class}/{image_basename}"

    image_filepath = os.path.join(root_path, image_class, image_basename)

    # If Callback was not specified, create a ProgressPercentage object
    if Callback is None:
        Callback = ProgressPercentage(
            root_path, image_class, image_id, image_ext, bucket_name,
            download=False
        )

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(
            image_filepath, bucket_name, image_object_name,
            Callback=Callback
        )
    except ClientError as e:
        logging.error(e)
        return False

    return True
