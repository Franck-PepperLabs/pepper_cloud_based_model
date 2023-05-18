from keras import Model
from keras.applications.mobilenet_v2 import MobileNetV2


def init_keras_model() -> Model:
    """Returns a MobileNetV2 model with the top layer removed
    
    Returns
    -------
    Model
        The initialized model.
    """
    # Create a MobileNetV2 model with pre-trained weights
    model = MobileNetV2(
        weights="imagenet",
        include_top=True,
        input_shape=(224, 224, 3)
    )
    
    # Set all layers in the model as non-trainable
    for layer in model.layers:
        layer.trainable = False
        
    # Create a new model without the top layer
    return Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )

