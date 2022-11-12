import time
from pprint import pprint

import ee


class ImageryDataset(ee.ImageCollection):
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

    def export_to_GS(self, test_split, task_name):
        sample_to_export = self.sample_imagery_L8SR(add_wsf_band=True)

        training = sample_to_export

        self.export_task = ee.batch.Export.image.toCloudStorage(
            image=sample_to_export,
            description=f"{task_name} - training",
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
        self.training_task.start()

    def monitor_export(self):

        while self.training_task.active():
            status = ee.data.listOperations()

            for i, dic in enumerate(status):
                if dic["name"].split("/")[-1] == self.training_task.id:
                    pprint(dic["metadata"])
                    print("-----------------------------------", "", sep="\n")
            time.sleep(60)
