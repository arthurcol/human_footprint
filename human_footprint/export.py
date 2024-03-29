import json

import ee

from human_footprint.imagery_dataset import ImageryDataset

if __name__ == "__main__":

    ee.Initialize()

    # with open(
    #     "/Users/arthurcollard/code/arthurcol/human_footprint/geometries/test_small.geojson",
    #     "rb",
    # ) as file:
    #     geo = json.load(file)

    france_borders = (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter('ADM0_NAME == "Belgium"')
        .geometry()
    )
    image = ImageryDataset(
        "LANDSAT/LC08/C02/T1_L2",
        area_of_interest=france_borders,
        date_of_interest=("2015-01-01", "2015-12-31"),
    )

    image.export_to_GS(task_name="belgium_2015/belgium_2015")
    image.monitor_export()
