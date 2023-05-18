from typing import Any, Union, Iterator

import io

import pandas as pd
import numpy as np

from PIL import Image

from keras.utils import img_to_array

from keras.applications.mobilenet_v2 import preprocess_input
from keras import Model

from pyspark.sql.functions import pandas_udf  # , PandasUDFType : Spark 3.4.0

from fruits.model import init_keras_model


def preprocess_img(content: Union[bytes, bytearray]) -> Any:
    """Pre-processes raw image bytes for prediction.

    Parameters
    ----------
    content : Union[bytes, bytearray]
        Raw image bytes.

    Returns
    -------
    Any
        Pre-processed image data.

    Raises
    ------
    ValueError
        If the provided content is not valid image bytes.
    """
    try:
        # Open the image from raw bytes and resize it
        img = Image.open(io.BytesIO(content)).resize([224, 224])
        # Convert the image to an array
        arr = img_to_array(img)
        return preprocess_input(arr)
    except Exception as e:
        raise ValueError(
            "Invalid image content. Please provide valid image bytes."
        ) from e


def extract_image_features(
    model: Model,
    content_series: pd.Series
) -> pd.Series:
    """Extracts image features from a pd.Series of raw images using the input
    model.

    Parameters
    ----------
    model : Model
        The pre-trained model used for feature extraction.
    content_series : pd.Series
        The pd.Series containing raw image data.

    Returns
    -------
    pd.Series
        The pd.Series containing the extracted image features.

    Note
    ----
    The function assumes that the `preprocess_img` function is defined separately.

    Raises
    ------
    ValueError
        If the model is not a valid TensorFlow Keras model.
    """
    try:
        # Preprocess the images in the content_series
        prep_imgs = np.stack(content_series.map(preprocess_img))
        # Extract the features using the model
        feats = model.predict(prep_imgs)
        # Flatten the feature tensors to vectors
        flat_feats = [f.flatten() for f in feats]
        return pd.Series(flat_feats)
    except Exception as e:
        raise ValueError(
            "Invalid model. Please provide a valid TensorFlow Keras model."
        ) from e


# See : https://www.databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html
@pandas_udf("array<float>")  # , PandasUDFType.SCALAR_ITER) = warning
def extract_image_features_udf(
    content_series_iter: Iterator[pd.Series]
) -> Iterator[pd.Series]:
    """This method is a Scalar Iterator pandas UDF wrapping our
    `extract_image_features` function. The decorator specifies that this
    returns a Spark DataFrame column of type ArrayType(FloatType).

    Parameters
    ----------
    content_series_iter : Iterator[pd.Series]
        An iterator over batches of data, where each batch is a pandas Series
        of image data.
    Yields
    ------
    Iterator[pd.Series]
        An iterator over the extracted image features for each batch of data.
    """
    # With Scalar Iterator pandas UDFs, we can load the model once and then
    # re-use it for multiple data batches. This amortizes the overhead of
    # loading big models.
    model = init_keras_model()
    model.set_weights(model_weights.values)
    for content_series in content_series_iter:
        yield extract_image_features(model, content_series)
