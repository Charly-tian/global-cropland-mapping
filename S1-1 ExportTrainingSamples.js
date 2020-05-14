/*
    This script is used to collect and export pixel-wise samples from GEE to Google Drive.
    
    We use 5°×5° grids to reduce the computation and work. In each grid, 50 random points
    were pre-generated in ArcGIS to ensure that every 2 points have a distance less than
    20km. To collect samples, we center each random point and obtain the neighbor 96×96 
    patch, thus obtain 193×193 image patches.

    Each image patch has 8 bands, of which 7 are feature bands, i.e., B, G, R, NIR, SWIR1,
    NDVI, DEM, the rest one is pixel-wise label from GFSAD30CE or GLC30.

    All the samples were exported to the Google Drive, and may also downloaded to local disks.

    Author: liuph0119
    Last Update: 2020-01-20
*/
var WorldGrid5d = ee.FeatureCollection("users/liuph/shape/WorldGrid5dC5"),
  worldRandomPoints50 = ee.FeatureCollection(
    "users/liuph/shape/WorldGrid5dRandomPoint50"
  ),
  worldBoundary = ee.FeatureCollection("users/liuph/shape/WorldBoundary");

var worldGrids = WorldGrid5d.filterMetadata(
  "centLat",
  "less_than",
  60
).filterMetadata("centLat", "greater_than", -60);

var liuph = require("users/liuph/default:utils/getSatCollections");
var vis = require("users/liuph/default:utils/vis");

var doExportGridSamples = function (
  gridId,
  grid,
  image,
  worldPoints,
  dst_folder,
  sel_vars
) {
  var geom = grid.geometry();
  var gridPoints = worldPoints.filterBounds(geom).filterBounds(worldBoundary);
  gridPoints = gridPoints.map(function (f) {
    return f.set("GridID", gridId);
  }); // set Grid ID for points

  var samples = image.sampleRegions({ collection: gridPoints, scale: 30 });
  // export the sample patch to Google Drive
  Export.table.toDrive({
    collection: samples,
    description: "" + gridId,
    folder: dst_folder,
    fileFormat: "TFRecord",
    selectors: sel_vars,
  });
};

// ########## main entry ##########
/* Landsat-8 + NDVI + Dem */
var lc08Image = liuph
  .getLC08SRMaskClouds(2015, 2015, 1, 12)
  .median()
  .multiply(0.0001)
  .select(["B2", "B3", "B4", "B5", "B6"], ["B", "G", "R", "NIR", "SWIR1"]);
var ndvi = lc08Image.normalizedDifference(["NIR", "R"]).rename("NDVI");
var dem = liuph.getDemSlope().select("Dem");
var satImage = lc08Image.addBands(ndvi).addBands(dem);

/* load cropland image and composite together */
var cropImage = ee
  .ImageCollection("users/gislfzhao/GFSAD30")
  .qualityMosaic("b1")
  .expression("b(0)==2?1:0")
  .rename("Cropland");
var compositeImage = satImage.addBands(cropImage);
// Map.addLayer(cropImage, {min: 0, max: 1, palette: vis.getPalette('Green')}, 'cropland', false);
// Map.addLayer(satImage,  {"opacity":1,"bands":["NIR","R","G"],"min":-0.2,"max":0.48090000000000005,"gamma":1}, 'sat', false);

/* get the neighborhood image patch of each pixel */
var imageNeighborhood = compositeImage.neighborhoodToArray({
  kernel: ee.Kernel.rectangle(96, 96, "pixels"),
});

var gridIdList = worldGrids
  .reduceColumns(ee.Reducer.toList(), ["ID"])
  .get("list")
  .getInfo();
print("Grid size: " + gridIdList.length);

for (var i = 0; i < 1; i++) {
  print(i + ":" + gridIdList[i]);
  var feature = worldGrids.filterMetadata("ID", "equals", gridIdList[i]);
  doExportGridSamples(
    gridIdList[i],
    feature,
    imageNeighborhood,
    worldRandomPoints50,
    "lc08sr_192x192",
    ["ID", "GridID", "B", "G", "R", "NIR", "SWIR1", "NDVI", "Dem", "Cropland"]
  );
}
// ########## end of main entry ##########
