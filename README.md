# City perimeter detection

Automatically extract the perimeter of a city given an aerial or satellite image of the city. A convolutional neural network is used to identify the road network in the image, and then image processing techniques expand the road network into a contour of the city.

![Summary diagram](https://github.com/carMartinez/city_perimeter_detect/blob/master/img/web/summary.png)

A full overview of this project including results and a summary of the methods
are [available here](http://cmartinez.io/perimeter-detection/). The basic workflow is listed below with each step
corresponding to a Jupyter notebook. More details are also available within the
notebooks. Note that the data used is not hosted here because of size and ownership,
but the procedure for freely downloading is explained in the first notebook.

1. [download.ipynb](https://github.com/carMartinez/city_perimeter_detect/blob/master/download.ipynb)
Download aerial images for 6 cities in Arizona. Aerial images are sourced
from the [National National Agriculture Imagery Program (NAIP) data available
on an AWS S3 bucket](https://docs.opendata.aws/aws-naip/readme.html). Many image files must downloaded to cover a single city resulting
in about 3-4 GB of data.
2. [label_roads.ipynb](https://github.com/carMartinez/city_perimeter_detect/blob/master/label_roads.ipynb)
Generate labels for machine learning by overlaying freely available Open
Street Map roads onto the georeferenced aerial images. Here we also store data
from the largest city (Phoenix, AZ) into an HDF5 dataset that will be used as
the training/development/test sets.
3. [preprocess.ipynb](https://github.com/carMartinez/city_perimeter_detect/blob/master/preprocess.ipynb)
Mosaic the multiple images file downloaded in step 1 so that there exists one
large aerial image per city. This will be easier to work with downstream.
4. [learn.ipynb](https://github.com/carMartinez/city_perimeter_detect/blob/master/learn.ipynb)
Train a convolutional neural network on the training set established in step 2.
A modified U-Net architecture is used.
5. [predict.ipynb](https://github.com/carMartinez/city_perimeter_detect/blob/master/predict.ipynb)
Apply the CNN to the 5 other cities to generate road predictions.
6. [morph.ipynb](https://github.com/carMartinez/city_perimeter_detect/blob/master/morph.ipynb)
Starting with the road network predictions, apply morphological operations
and a contour extraction to produce a bounding
polygon around the road network. This is taken to be the city perimeter.

Results for the 5 cities are shown below.

![Results](https://github.com/carMartinez/city_perimeter_detect/blob/master/img/web/results.png)
