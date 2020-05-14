import pandas as pd
import numpy as np
import ogr
import os
from osgeo import gdal
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def read_validation_set(filename):
    """ read features from a shapefile file """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(filename, 0)
    assert ds is not None, "data source is empty!"

    layer = ds.GetLayer(0)
    n_features = layer.GetFeatureCount()
    features = [layer.GetFeature(i) for i in range(n_features)]
    print("feature count:", len(features))
    ds.Destroy()
    return features


def world2pixel(ds, x, y):
    """ Calculate the row and row index of a point in an image.

    # Args:
        ds (RasterDataset): .
        x (float): Longitude of the point.
        y (float): Latitude of the point.

    # Returns:
        pixel (int), line (int): the col and row index of a specific pixel
    """
    geom_transform = ds.GetGeoTransform()
    up_left_x = geom_transform[0]
    up_left_y = geom_transform[3]
    pixel_size_x = geom_transform[1]
    pixel_size_y = - geom_transform[5]
    pixel = int((x - up_left_x) / pixel_size_x)
    line = int((up_left_y - y) / pixel_size_y)
    return pixel, line


def generate_overlay_table(val_points_filename, tif_filenames, table_save_path):
    val_points = read_validation_set(val_points_filename)
    tif_datasets = []
    for tif_filename in tif_filenames:
        tif_datasets.append(gdal.Open(tif_filename, gdal.GA_ReadOnly))
    n_bands = tif_datasets[0].RasterCount
    results = []

    for point in tqdm(val_points):
        ID = point.GetFieldAsInteger('ID')
        lng = point.GetFieldAsDouble('X')
        lat = point.GetFieldAsDouble('Y')
        gt = point.GetFieldAsInteger('gt2015')

        res = OrderedDict()
        for ds, tif_filename in zip(tif_datasets, tif_filenames):
            _pixel, _line = world2pixel(ds, lng, lat)
            if (0 <= _pixel < ds.RasterXSize) and (0 <= _line < ds.RasterYSize):
                tif_values = ds.ReadAsArray(
                    _pixel, _line, 1, 1).squeeze().tolist()
                if not isinstance(tif_values, list):
                    tif_values = [tif_values]

                # print(ID, lng, lat, gt, _pixel, _line, tif_values, tif_filename)
                res['PointID'] = ID
                res['Pointlng'] = lng
                res['PointLat'] = lat
                res['gt'] = gt
                for i in range(n_bands):
                    res['b_%d' % (i+1)] = tif_values[i]
                results.append(res)
                break
    pd.DataFrame.from_dict(results, orient='columns').to_csv(
        table_save_path, index=False)

    del tif_datasets


def evaluate_accuracy_report(overlay_table_filename, acc_filename, gt_col_name='gt', pred_col_name='b_1'):
    df = pd.read_csv(overlay_table_filename)
    y_true = df[[gt_col_name]].values
    y_true = np.where(y_true == 10, 1, 0)
    y_pred = df[[pred_col_name]].values
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    metrics = OrderedDict()
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['cm'] = confusion_matrix(y_true, y_pred).tolist()
    print(metrics)
    pd.DataFrame.from_dict([metrics]).to_csv(
        acc_filename, index=False, float_format='%.6f')


if __name__ == '__main__':
    gt_filename = '/WorldValidationPoints.shp'
    preds_dir = "G:/crop/images/preds/bcegdice_pspnet50_2015lc08srsmote_c1_192x192x7"
    model_version = 'bcegdice_pspnet50_2015lc08srsmote_c1_192x192x7'
    acc_save_dir = 'G:/crop/images/accs/bcegdice_pspnet50_2015lc08srsmote_c1_192x192x7'

    os.makedirs(acc_save_dir, exist_ok=True)
    table_save_path = acc_save_dir + '/{}.csv'.format(model_version)
    acc_save_path = acc_save_dir + '/acc_{}.csv'.format(model_version)

    preds_filenames = [os.path.join(preds_dir, fn)
                       for fn in os.listdir(preds_dir)]
    generate_overlay_table(gt_filename, preds_filenames, table_save_path)
    evaluate_accuracy_report(overlay_table_filename=table_save_path,
                             acc_filename=acc_save_path,
                             gt_col_name='gt', pred_col_name='b_1')
