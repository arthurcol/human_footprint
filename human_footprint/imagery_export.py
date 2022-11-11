import ee
import os
import json
import time
from pprint import pprint

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]


def prepare_imagery(image):
    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select("pixel_qa")
    mask = (
        qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    )

    # geometry = load_geometry(os.getenv("REGION_GEOJSON"))
    # date_range = {"min": "2015-01-01", "max": "2015-12-31"}
    # bound_filter = ee.Filter.bounds(geometry)
    # date_filter = ee.Filter.date(date_range["min"], date_range["max"])

    # image = image.filterBounds(geometry).filterDate("2015-01-01", "2015-12-31")

    return image.updateMask(mask).select(BANDS).divide(1000)


def load_geometry(region_name):
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "geometries", region_name
    )
    with open(path, "rb") as file:
        geometry = json.load(file)

    return geometry["features"][0]["geometry"]


def create_wsf_featurecollection(country, scale):
    img = ee.Image(
        "DLR/WSF/WSF2015/v1"
    ).unmask()  # unmask needed to make zeros appear as properties for settlement
    country = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
        ee.Filter.eq("country_na", country)
    )

    fromList = ee.List([255])
    toList = ee.List([1])

    img = img.remap(fromList, toList, defaultValue=0, bandName="settlement").rename(
        "settlement"
    )

    vect = img.reduceToVectors(
        geometry=country,
        # crs=img.projection(),
        scale=scale,
        # geometryType= 'polygon',
        eightConnected=False,
        labelProperty="settlement",
        # reducer= ee.Reducer.mean()
    )
    return vect


if __name__ == "__main__":
    ee.Initialize()

    ## load the imagery
    geo = load_geometry(os.getenv("REGION_GEOJSON"))
    image_landsat = ee.ImageCollection(os.getenv("IMAGE_NAME"))
    sample_imagery = (
        image_landsat.filterBounds(geo)
        .filterDate("2015-01-01", "2015-12-31")
        .map(prepare_imagery)
        .median()
    )

    ## create WSF  FeatureCollection
    wsf_fc = create_wsf_featurecollection(country="France", scale=1000)

    # link image and fc
    sample_data = sample_imagery.sampleRegions(
        collection=wsf_fc, properties=["settlement"], scale=20
    )

    ##create export tasks
    task_test = ee.batch.Export.table.toCloudStorage(
        collection=sample_data,
        description="test export",
        fileNamePrefix="gs://human-footprint/test_sample.tfrecord.gz",
        bucket="human-footprint",
        fileFormat="TFRecord",
    )

    task_test.start()

    while task_test.active():
        print(f"Polling for task (id: {task_test.id}).")
        status = ee.data.listOperations()
        for i, dic in enumerate(status):
            if dic["name"].split("/")[-1] == task_test.id:
                pprint(dic)
        time.sleep(60)
    print("Done !")
