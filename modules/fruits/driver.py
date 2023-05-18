from typing import Any, Optional
import os

from keras import Model

import pyspark.sql as pss
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import element_at, split

from pepper.env import get_project_dir

# Set globally by driver and read locally by each executor
broadcast_weights = None

def init_spark_session() -> SparkSession:
    """Initializes and returns a Spark session.

    Returns
    -------
    SparkSession
        The initialized Spark session.
    
    Note
    ----
    When you call getOrCreate() on a SparkSession instance, Spark will either
    retrieve an existing session if it already exists in the current JVM (Java
    Virtual Machine), or create a new session if no session exists. This
    behavior ensures that there is only one instance (singleton) of the Spark
    session per JVM, allowing efficient resource utilization.
    """
    # Set the Spark master URL to run locally with all available cores
    # Note: 'local[*]' utilizes all available cores on the local machine
    # for parallel execution of Spark tasks.
    spark_master_url = "local[*]"

    # Set the configuration option for writing Parquet files in the legacy format
    # This is useful when interacting with older systems that do not support
    # the new Parquet format introduced in Spark 2.0.
    spark_config = {
        "spark.sql.parquet.writeLegacyFormat": "true"
    }

    # Create the Spark session instance if it doesn't exist, or retrieve the existing one
    return (
        SparkSession
        .builder
        .appName("Fruits")  # Set the application name
        .master(spark_master_url)  # Set the Spark master URL
        # see : https://spark.apache.org/docs/latest/api/python/reference/
        #       pyspark.sql/api/pyspark.sql.SparkSession.builder.config.html
        # You cannot do directly the pythonic **spark_config
        .config(map=spark_config)  # Set additional configuration options
        .getOrCreate()  # Get or create the Spark session
    )

# Get the Spark session context
# sc = spark.sparkContext

def load_images(spark: SparkSession):
    raw_src_im_dir = os.path.join(
        get_project_dir(),
        r"dataset\fruits-360_dataset\Test"
    )
    images = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(raw_src_im_dir)
    )
    
    # Add a 'label' column based on the directory structure
    return images.withColumn(
        "label",
        element_at(split(images["path"], "/"), -2)
    )
    
    
def load_images(
    image_path: Optional[str] = None
) -> pss.DataFrame:
    """Loads images from a directory and returns a DataFrame.

    Parameters
    ----------
    image_path : str, optional
        The path to the directory containing the images. If not specified, 
        the default directory 'dataset/fruits-360_dataset/Test' relative to
        the project directory will be used.

    Returns
    -------
    DataFrame
        A DataFrame containing the loaded images.

    Note
    ----
    This function uses the SparkSession instance obtained from
    init_spark_session().

    The images are loaded from the specified directory.
    The directory can contain image files in various formats.

    The 'binaryFile' format is used to read the images as binary files.

    The options 'pathGlobFilter' and 'recursiveFileLookup' are set to filter
    the files and enable recursive file lookup, respectively.

    The resulting DataFrame contains the loaded images with an additional
    'label' column.
    """
    # Obtain the SparkSession instance
    spark = init_spark_session()

    # Set the path to the directory containing the images
    if image_path is None:
        image_path = os.path.join("dataset", "fruits-360_dataset", "Test")
    raw_src_im_dir = os.path.join(get_project_dir(), image_path)

    # Read the images using the 'binaryFile' format and specified options
    images = (
        # Get a `pyspark.sql.readwriter.DataFrameReader`
        spark.read.format("binaryFile")
        # Filter files with the '*.jpg' extension
        .option("pathGlobFilter", "*.jpg")
        # Enable recursive file lookup
        .option("recursiveFileLookup", "true")
        # Load images from the specified directory
        .load(raw_src_im_dir)
        # Return it as a `pyspark.sql.DataFrame`
    )

    # Extract the label from the directory structure using the 'element_at'
    # and 'split' functions
    
    # Split the path by '/'
    path_elements = split(images["path"], "/")
    
    # Get the second-to-last element as the label
    label = element_at(path_elements, -2)

    # Add `label` as the 'label' column to the DataFrame, and return it
    return images.withColumn("label", label)



# Sans faire de TP, je suis mort, rien n'est clair
def broadcast_model_weights(spark: SparkSession, model: Model):
    """broadcasts the pre-trained weights to the cluster's executors.
    
    Parameters
    ----------
    sc : SparkContext
        The Spark context instance.
    """
    # Broadcast the weights of the new model to all Spark workers
    global broadcast_weights
    sc = spark.sparkContext
    broadcast_weights = sc.broadcast(model.get_weights())
    