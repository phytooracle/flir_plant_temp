# FLiR Plant Temperature
This script detects individual plants and extracts their canopy temperature from thermal imagery.

## Inputs
Directory containing GeoTIFs.

## Outputs
The script outputs a single CSV file listing all plant detections, including their geocoordinates, bounding area, and canopy temperature.

## Arguments and Flags
* **Positional Arguments:**
    * **Directory containing geoTIFFs:** 'dir'
      
* **Required Arguments:**
   * **Object detection model:** '-m', '--model'
   * **GeoJSON containing plot boundaries:** '-g', '--geojson'
   * **Date of data collection:** '-d', '--date'

* **Optional Arguments:**
   * **Output directory:** '-od', '--outdir'
   * **Output filename:** '-of', '--outfile'
   * **Central processing units (CPUs):**  '-c', '--cpu'
