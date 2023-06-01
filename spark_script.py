import pandas as pd
import numpy as np
import io
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model

from PIL import Image
from pyspark.sql.functions import col, pandas_udf, element_at, split
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import PCA as spa_PCA
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector, DenseVector


PATH = "s3://ds-p8"
PATH_DATA = PATH + "/data"
PATH_RESULT = PATH + "/Results"

k_components = 138

# Building Spark session and context
spark = (
    SparkSession.builder
    .appName("ds_p8")
    .config("spark.sql.parquet.writeLegacyFormat", "true")
    .getOrCreate()
)

sc = spark.sparkContext


# Loading images from PATH_DATA
images = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.jpg")
    .option("recursiveFileLookup", "true")
    .load(PATH_DATA)
)

images = images.withColumn("label", element_at(split(images["path"], "/"), -2))

# model setup :
model = MobileNetV2(
    weights="imagenet",
    include_top=True,
    input_shape=(224, 224, 3)
    )

new_model = Model(
    inputs=model.input,
    outputs=model.layers[-2].output
    )

# Broadcasting the weights to the workers :

brodcast_weights = sc.broadcast(new_model.get_weights())


def model_fn():
    """
    Returns a MobileNetV2 model with top layer removed
    and broadcasted pretrained weights.
    """
    model = MobileNetV2(
        weights="imagenet",
        include_top=True,
        input_shape=(224, 224, 3)
        )

    for layer in model.layers:
        layer.trainable = False

    new_model = tf.kerras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output
        )

    new_model.set_weights(brodcast_weights.value)
    return new_model


def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.

    Returns:
    - pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)


@pandas_udf(ArrayType(FloatType()))
def featurize_udf(content_series_iter):
    """
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    Args:
    - content_series_iter: This argument is an iterator over batches of data, where each batch
    is a pandas Series of image data.
    """
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.

    model = model_fn()

    for content_series in content_series_iter:
        yield featurize_series(model, content_series)


# Extracting features :
features_df = images.repartition(24).select(
    col("path"),
    col("label"),
    featurize_udf("content").alias("features")
    )

# PCA over extracted features
# Converting arrays into vectors, spark's PCA expects that format
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

df_spark_vector = features_df.select(
    features_df["path"],
    features_df["label"],
    list_to_vector_udf(features_df["features"]).alias("features")
)

spark_pca = spa_PCA(k=k_components, inputCol="features")
spark_pca.setOutputCol("reduced_features")

spark_pca_model = spark_pca.fit(df_spark_vector)

df_spark_vector = spark_pca_model.transform(dataset=df_spark_vector)


# Converting vectors back to arrays for readability
vector_to_array_udf = udf(
    lambda vector: vector.toArray().tolist()
    if isinstance(vector, (DenseVector, SparseVector))
    else vector,
    ArrayType(FloatType())
)

df_spark_vector = df_spark_vector.withColumn("features", vector_to_array_udf(df_spark_vector["features"]))
df_spark_vector = df_spark_vector.withColumn(
    "reduced_features", vector_to_array_udf(df_spark_vector["reduced_features"])
    )

# Saving as parquet
df_spark_vector.write.mode("overwrite").parquet(PATH_RESULT)
