import json
import time
from pprint import pprint
from typing import Dict, List

import ee
import tensorflow as tf
from google.cloud import storage


class ImageryDataset(ee.ImageCollection):
    """
    Instantiate an object able to generate Earth Enfine images and launch tasks
    to export them to GCS.
    Inherits from `earthengine.ImageCollection`
    """

    def __init__(self, *args, date_of_interest, area_of_interest):
        assert (
            isinstance(date_of_interest, tuple) and len(date_of_interest) == 2
        ), "date_of_interest should be a tuple of date_min, date_max"
        assert isinstance(
            area_of_interest, (ee.Geometry)
        ), "area_of_interest should be a geometry object"

        self.doi = date_of_interest
        self.aoi = area_of_interest
        super().__init__(
            *args,
        )

    @staticmethod
    def prepare_imagery_L8SR(image):
        qa_mask = image.select("QA_PIXEL").bitwiseAnd(int("11111", 2)).eq(0)
        sat_mask = image.select("QA_RADSAT").eq(0)

        def get_scaling_factor(factor_names):
            factor_list = image.toDictionary().select(factor_names).values()
            return ee.Image.constant(factor_list)

        scale_values = get_scaling_factor(
            ["REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B10"]
        )
        offset_values = get_scaling_factor(
            ["REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B10"]
        )

        scaled = image.select("SR_B.|ST_B10").multiply(scale_values).add(offset_values)

        return (
            image.addBands(scaled, None, True).updateMask(qa_mask).updateMask(sat_mask)
        )

    def sample_imagery_L8SR(self, add_wsf_band=True):
        wsf_image = (
            ee.Image("DLR/WSF/WSF2015/v1")
            .unmask()
            .remap([255, 0], [1, 0])
            .rename("settlement")
        )
        wsf_band = wsf_image.select(["settlement"])
        wsf_band = wsf_band.cast({"settlement": "int64"})
        sample = (
            self.filterDate(self.doi[0], self.doi[1])
            .filterBounds(self.aoi)
            .map(self.prepare_imagery_L8SR)
            .select(["SR_B2", "SR_B3", "SR_B4"])
            .median()
        )
        if add_wsf_band:
            return sample.addBands(wsf_band)
        return sample

    def export_to_GS(self, task_name):
        sample_to_export = self.sample_imagery_L8SR(add_wsf_band=True)

        self.export_task = ee.batch.Export.image.toCloudStorage(
            image=sample_to_export,
            description=f"{task_name.split('/')[0]}",
            bucket="human-footprint",
            fileNamePrefix=task_name,
            fileFormat="TFRecord",
            maxPixels=1e13,
            scale=30,
            region=self.aoi,
            formatOptions={
                "patchDimensions": [256, 256],
                "compressed": True,
            },
        )
        self.export_task.start()

    def monitor_export(self):

        while self.export_task.active():
            status = ee.data.listOperations()

            for i, dic in enumerate(status):
                if dic["name"].split("/")[-1] == self.export_task.id:
                    pprint(dic["metadata"])
                    print("-----------------------------------", "", sep="\n")
            time.sleep(60)


class ImageryDatasetParser:  # TODO check
    """
    Instantiate a function to map over a `tf.data.Dataset` to be able to parse
    the SR images loaded into tfrecords file by `ImageryDataset`
    """

    def __init__(self, mixer_file, band_names, label):
        self.patch_width = mixer_file["patchDimensions"][0]
        self.patch_height = mixer_file["patchDimensions"][1]
        self.patches = mixer_file["totalPatches"]
        self.band_names = band_names
        self.label = label
        self.image_columns = [
            tf.io.FixedLenFeature(
                shape=[self.patch_width, self.patch_height], dtype=tf.float32
            )
            for k in self.band_names
        ] + [
            tf.io.FixedLenFeature(
                shape=[self.patch_width, self.patch_height, 1], dtype=tf.int64
            )
        ]

        self.image_features_dict = dict(
            zip(self.band_names + [self.label], self.image_columns)
        )

    def __call__(self, example_proto):
        parsed_features = tf.io.parse_example(example_proto, self.image_features_dict)
        label = parsed_features.pop(self.label)

        # B4 = red ; B3 = green ; B2 = blue
        img = tf.stack(
            [parsed_features[band] for band in self.band_names],
            -1,
        )

        return img, label


def download_gcs(bucket_name: str, blob_name: str) -> str:
    """Helper function to download files from GCS buckets, files being returned
    as string.

    Parameters
    ----------
    bucket_name : str
        Name of the bucket the file is in.
    blob_name : str
        name of the file. / indicates a path

    Returns
    -------
    str
        file as string - will need to be decoded
    """
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    contents = blob.download_as_string()
    return contents


def get_mixer_json(bucket_name, blob_name) -> Dict:
    """Wrapper function to specifically download a json, using `download_gc`
    function.

    Parameters
    ----------
    bucket_name : str
        Name of the bucket the file is in.
    blob_name : str
        name of the file. / indicates a path

    Returns
    -------
    Dict
    """
    content = download_gcs(bucket_name, blob_name)
    return json.loads(content)


def list_blobs(
    bucket_name: str,
    prefix: str = None,
    delimiter: str = None,
    extension: str = "tfrecord.gz",
) -> List:
    """List all files in a bucket, possibly filtered with extension.

    Parameters
    ----------
    bucket_name : str
        Name of the bucket the files are in
    prefix : str, optional
        prefix of the files, / indicates a path if specified in the delimiter `
        argument, by default None
    delimiter : str, optional
        character to be interpreted as a path delimiter, if present in the
        prefix, by default None
    extension : str, optional
        If provided, the function returns only files with this extension,
        by default "tfrecord.gz"

    Returns
    -------
    List
        List of all file names meeting given conditions.
    """

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    file_names = [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(extension)
    ]
    return file_names
