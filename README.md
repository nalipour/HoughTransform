# Hough Transform

A package for track reconstruction using the Hough Transformation.
Developed for the tracking of the drift chamber of the FCCeeIDEA detector concept.

## Installation (for Mac OS X)

* Clone the repository: `git clone https://github.com/nalipour/HoughTransform.git`
* Install python: `brew install python3`
* Create a virtual environment *my-env*, install the required packages and run the program

  * `virtualenv --python python3 my-env`
  * `source my-env/bin/activate`
  * `pip install -r requirements.txt `

## Run

First, the output hits from the FCCSW simulation needs to be written in a CSV file format using the command:

 `python rootTree2CSV.py --input=hits.root --output = hits.csv`


The Hough transform can be run using the following command:

`python main.py --input=hits.csv --output=./plot/`


## The Hough transform for one particle track
![your_image_name](images/zoom_HT_withMax.png)
