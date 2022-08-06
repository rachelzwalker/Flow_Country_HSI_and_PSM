import pandas as pd
import geopandas
import rasterio as rio
import rasterio.features
import rasterio.warp


def output_csv_for_analysis(data_tiff, geom_file, name):
    dataframe = point_id(data_tiff)
    geometry_file = pd.read_csv(geom_file)

    data_gdf = pd.merge(dataframe, geometry_file, on="POINTID")

    data_gdf = data_gdf.rename(columns={"wkt_geom": "geometry"})

    data_gdf['geometry'] = geopandas.GeoSeries.from_wkt(data_gdf['geometry'])

    gdf = geopandas.GeoDataFrame(data_gdf, geometry='geometry')

    gdf.to_csv(
        f"{name}.csv")
    gdf.to_file(
        f"{name}.shp")


# make shapefile in GIS converting raster to vector using one of the bands

def point_id(data_tif):
    dataframe = make_dataframe(data_tif)
    dataframe["POINTID"] = ""
    index_num = 0
    point_num = 1

    for index, row in dataframe.iterrows():
        dataframe.iloc[index_num, 358] = point_num
        point_num += 1
        index_num += 1

    return dataframe


def make_dataframe(data_tif):
    data = rio.open(data_tif)
    array = data.read()
    dataframe = pd.DataFrame()

    array_num = 0
    band_num = 1

    for num in range(358):
        dataframe['band' + str(band_num)] = array[array_num].ravel()
        array_num += 1
        band_num += 1
    return dataframe



def import_with_rasterio(data_tif):
    with rio.open(data_tif) as dataset:
        # Read the dataset's valid data mask as a ndarray.
        mask = dataset.dataset_mask()

        # Extract feature shapes and values from the array.
        for geom, val in rasterio.features.shapes(
                mask, transform=dataset.transform):
            # Transform shapes from the dataset's own coordinate
            # reference system to CRS84 (EPSG:4326).
            geom = rasterio.warp.transform_geom(
                dataset.crs, 'EPSG:32630', geom, precision=6)
    return geom

