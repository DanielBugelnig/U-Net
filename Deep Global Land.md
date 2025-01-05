#### DEEP GLOBAL LAND DATA SET LINK

https://www.dropbox.com/scl/fi/0xustv7gy1icnrmyjhax3/Deep-Globe-Land.zip?rlkey=b2fwucpsnf1jy61j97c2wxjww&st=joknlcl0&dl=0

- This link is to download the data set for the project

### It contains:
  - The training data for Land Cover Challenge contains 560 satellite imagery in RGB, size 2448x2448.
- The imagery has 50cm pixel resolution, collected by DigitalGlobe's satellite.
- The dataset contains 121 validation and 121 test images.

### Label:
- Each satellite image is paired with a mask image for land cover annotation. The mask is a RGB image with 7 classes of labels, using color-coding (R, G, B) as follows.

    - Urban land: 0,255,255 - Man-made, built up areas with human artifacts (can ignore roads for now which is hard to label)
    - Agriculture land: 255,255,0 - Farms, any planned (i.e. regular) plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations.
    - Rangeland: 255,0,255 - Any non-forest, non-farm, green land, grass
    - Forest land: 0,255,0 - Any land with x% tree crown density plus clearcuts.
    - Water: 0,0,255 - Rivers, oceans, lakes, wetland, ponds.
    - Barren land: 255,255,255 - Mountain, land, rock, dessert, beach, no vegetation
    - Unknown: 0,0,0 - Clouds and others
- File names for satellite images and the corresponding mask image are id _sat.jpg and id _mask.png. id is a randomized integer.
- The values of the mask image may not be pure 0 and 255. When converting to labels, please binarize them at threshold 128.
  
  
