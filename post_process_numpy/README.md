# Version

Run with Python 3.10.2

# Dependencies

Install the latest version using pip or conda

##### Numpy==1.22.3
https://pypi.org/project/numpy/

##### Pandas==1.4.2
https://pypi.org/project/pandas/

##### Plotly==5.6.0
https://pypi.org/project/plotly/

##### Scipy==1.8.0
https://pypi.org/project/scipy/

##### Sklearn==1.0.2
https://pypi.org/project/scikit-learn/

# Code

This code is used to take in points distributed from planes given in different formats, run the points through PCA to estimate the planes that the points are
distributed from, and find intersections between the planes.  Additionally, provides a way to visualize the planes and give the angles between planes to check
the accuracy of the plane estimations.

## Plane Dataclass

A class for representing a plane in standard form, ax + by + cz + d = 0.

### Methods

##### __post_init__(self) -> None

Post init method for declaring variables from the above dataclass fields.

##### __str__(self) -> str

Overide of the default dataclass str method to print nicely.

##### randomPoint(self, n: float, b: float) -> list[float]

Generates a point in R3 that is close to
the plane, but within the closed boundary,
[-b, b] with some added uniform noise, n.

##### getMeshGrids(self, term: str, axisLengths: list[float]) -> tuple[np.meshgrid, np.meshgrid, np.meshgrid]

Takes as input a term, either 'x', 'y', 'z' then calculates
and returns a tuple of 3 mesh grids which can be used for
graphing the plane in terms of term using the axis lengths
in the axisLengths tuple for x, y, and z respectively.

### Static Methods

##### estimatePlane(points: np.ndarray) -> "Plane":

Takes as input a matrix which is the number of
points by 3 as a coordinate in R3 has 3 components,
then fits and returns the bestfit plane using PCA.

##### solveIntersection(planes: np.ndarray, threshold: float=100) -> np.ndarray:

Takes in a list of planes, planes, then calculates the
point where 3 planes intersect if any exists. Removes all
points with at least one component larger than the threshold.

##### anglesBetweenPlanes(planes: list["Plane"], planeNames: list[str], radians: bool=True) -> list[tuple[str, str, float]]:

Takes as input a list of planes, planes, and a parallel list
of strings, planeNames, representing the name of each plane in
planes, then calculates and returns the a list of tuples
containing the names of the two corresponding planes and the
angle between them.


## Functions

##### parsePoints1(headerFileName: str, constantTransform: np.ndarray, sampleSize: int) -> tuple[np.ndarray, list[str]]

Takes in a filename specifying a header file, headerFileName, that
is properly formatted and a constant transform, constantTransform,
from the probe to the end effector and calculates and returns a
list of all the points for each plane specified in the header
file.  The return is a 3D tensor which is number of planes by
number of points by 3 since each coordinate has 3 components as
they are in R3.  Ad hoc for this problem and the header file
provided.  If N Points are specified in the header, then
sampleSize points are sampled for each plane.

##### parsePoints2(fileNames: list[str], sampleSize: int,threshold: int) -> tuple[list[list[float]], list[str]]

Takes in a list of filename specifying csv files that are properly
formatted and returns a list of all the points for each plane,
i.e., on for each item in the list fileNames.  The return is a 3D
tensor which is number of planes by number of points by 3 since
each coordinate has 3 components as they are in R3.  Ad hoc for
this problem and the csvs provided.  If N Points are specified
in the csvs, then sampleSize points are sampled for each plane.
Removes any points with any components above threshold.

##### visualizeResults(planes: list[Plane], intersections: np.ndarray, terms: list[str], axisLengths: list[float], planePoints: np.ndarray, graphPlanes: bool=True, graphPoints: bool=True, graphIntersections: bool=True) -> None

Visualize the planes, ax + by + cz + d = 0, their
intersection points, and the points that were used
to find the fit planes in 3D to evaluate the results.
3 boolean flags for what to graph are also taken as input.

##### run(n: int = math.inf) -> None

The main function used for all execution and to not pollute the
global namespace. Parts (A) - (F) define all parts of the code.
Takes in a variable to n for sampling points.  Uses all points
if n = math.inf (default).  n != 0 Works for parsePoints1(),
however it is NOT fully implemented for parsePoints2() as it
does not sample points.

