import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dataclasses import dataclass, field
from itertools import chain, combinations
from random import uniform, sample
from scipy.spatial.transform import Rotation
from sklearn import decomposition


@dataclass
class Plane():
    """
    A class for representing a plane in
    standard form, ax + by + cz + d = 0.
    """
    a: float  # Coefficient of x.
    b: float  # Coefficient of y.
    c: float  # Coefficient of z.
    d: float
    planeList: np.ndarray = field(init=False)
    norm: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """
        Post init method for declaring variables
        from the above dataclass fields.
        """

        # Initialize a list of all the coorindates of the
        # plane and the normal vector associated with the plane.
        self.planeList = np.array([self.a, self.b, self.c, self.d])
        self.norm = np.array([self.a, self.b, self.c])

    def __str__(self) -> str:
        """Overide of the default dataclass str method."""
        p = 10  # Precision after the decimal.

        # Record the signs of the coefficients
        # for a pretty str method.
        signs = [('+' if c >= 0 else '-') for c in self.planeList]
        if signs[0] == '+':
            signs[0] = ' '  # Lineup equations starting with '-'.

        # Initalize the format string and truncate
        # the values of the coefficients to the specified
        # precision, removing leading negative signs.
        formatStr = "{}{}x {} {}y {} {}z {} {} = 0"
        planeListStr = [(str(c)[1:p+3]
                        if c < 0
                        else str(c)[:p+2])
                        for c in self.planeList]

        # Return the formatted string with correct signs.
        return formatStr.format(*chain(*zip(signs, planeListStr)))

    def randomPoint(self, n: float, b: float) -> list[float]:
        """
        Generates a point in R3 that is close to
        the plane, but within the closed boundary,
        [-b, b] with some added uniform noise, n.
        """

        # Pick random points in the boundary and
        # solve for a point on the plane, (x, y, z).
        if self.c != 0:
            x, y = uniform(-b, b), uniform(-b, b)
            z = (-self.a/self.c)*x-(self.b/self.c)*y-(self.d/self.c)
        elif self.b != 0:
            x, z = uniform(-b, b), uniform(-b, b)
            y = (-self.a/self.b)*x-(self.c/self.b)*z-(self.d/self.b)
        elif self.a != 0:
            y, z = uniform(-b, b), uniform(-b, b)
            x = (-self.b/self.a)*y-(self.c/self.a)*z-(self.d/self.a)

        # Add noise to the 3 components and return
        # the point that is close to the given plane.
        return [x + uniform(-n, n),
                y + uniform(-n, n),
                z + uniform(-n, n)]

    def getMeshGrids(self,
                     term: str,
                     axisLengths: list[float]
                    ) -> tuple[np.meshgrid, np.meshgrid, np.meshgrid]:
        """
        Takes as input a term, either 'x', 'y', 'z' then calculates
        and returns a tuple of 3 mesh grids which can be used for
        graphing the plane in terms of term using the axis lengths
        in the axisLengths tuple for x, y, and z respectively.
        """

        # Define spans for each axis using axisLengths.
        spanx = np.linspace(axisLengths[0], axisLengths[1], 30)
        spany = np.linspace(axisLengths[2], axisLengths[3], 30)
        spanz = np.linspace(axisLengths[4], axisLengths[5], 30)

        # Calculate the plane equation
        # and record the plane to be drawn.
        if term == 'z':
            x, y = np.meshgrid(spanx, spany)
            z = (-self.a/self.c)*x-(self.b/self.c)*x.T-(self.d/self.c)
        elif term == 'y':
            x, z = np.meshgrid(spanx, spanz)
            y = (-self.a/self.b)*x-(self.c/self.b)*z-(self.d/self.b)
        elif term == 'x':
            y, z = np.meshgrid(spany, spanz)
            x = (-self.b/self.a)*y-(self.c/self.a)*z-(self.d/self.a)

        return x, y, z  # Return the three meshgrids for x, y, and z.

    @staticmethod
    def estimatePlane(points: np.ndarray) -> "Plane":
        """
        Takes as input a matrix which is the number of
        points by 3 as a coordinate in R3 has 3 components,
        then fits and returns the bestfit plane using PCA.
        """

        # Calculate the mean point
        # and normalize all the points.
        m = np.average(points, axis=0)
        points -= m

        # Set up PCA for 3 components, the vectors spanning
        # the plane and the normal vector, then fit the
        # points using PCA, and record the plane.
        pca = decomposition.PCA(n_components=3)
        pca.fit(points)
        # print(pca.singular_values_)
        # print(pca.components_)
        return Plane(*pca.components_[2, :],
                     -pca.components_[2, :] @ m)

    @staticmethod
    def solveIntersection(planes: np.ndarray,
                          threshold: float=100
                         ) -> np.ndarray:
        """
        Takes in a list of planes, planes, then calculates the
        point where 3 planes intersect if any exists. Removes all
        points with at least one component larger than the threshold.
        """

        # Get all possible combinations of 3 planes.
        possibleIntersections = combinations(planes, 3)
        intersections = []  # A list to store all intersection points.

        # Loop for all possible sets of intersecting planes.
        for p1, p2, p3 in possibleIntersections:

            # Try and find the intersection of the 3 planes.
            try:
                a = np.array([[p1.a, p1.b, p1.c],
                              [p2.a, p2.b, p2.c],
                              [p3.a, p3.b, p3.c]])
                b = np.array([p1.d, p2.d, p3.d])

                point = np.linalg.solve(a, b)  # Gaussian Elimination.

                # Don't record points intersecting very far away.
                if any([abs(component) > threshold
                        for component in point]):
                    continue

                intersections.append(point)  # Record intersection.

            except np.linalg.LinAlgError:

                # Pass since a singular matrix is found, i.e.,
                # there is no solution or many solutions.
                pass

        return np.array(intersections)  # Return all intersections.

    @staticmethod
    def anglesBetweenPlanes(planes: list["Plane"],
                            planeNames: list[str],
                            radians: bool=True
                           ) -> list[tuple[str, str, float]]:
        """
        Takes as input a list of planes, planes, and a parallel list
        of strings, planeNames, representing the name of each plane in
        planes, then calculates and returns the a list of tuples
        containing the names of the two corresponding planes and the
        angle between them.
        """

        namesAngles = []  # Instantiate a container to return.

        # Loop for all pairs of planes.
        pairs = combinations(zip(planeNames, planes), 2)
        for (n1, p1), (n2, p2) in pairs:

            # Calculate the angle between
            # planes p1 and p2 in correct unit.
            angle = np.arccos(-abs(p1.norm @ p2.norm))
            angle = angle if radians else angle*(180/np.pi)

            # Record the tuple for this pair.
            namesAngles.append((n1, n2, angle))

        return namesAngles  # Return the list of tuples.


def parsePoints1(headerFileName: str,
                 constantTransform: np.ndarray,
                 sampleSize: int,
                ) -> tuple[np.ndarray, list[str]]:
    """
    Takes in a filename specifying a header file, headerFileName, that
    is properly formatted and a constant transform, constantTransform,
    from the probe to the end effector and calculates and returns a
    list of all the points for each plane specified in the header
    file.  The return is a 3D tensor which is number of planes by
    number of points by 3 since each coordinate has 3 components as
    they are in R3.  Ad hoc for this problem and the header file
    provided.  If N Points are specified in the header, then
    sampleSize points are sampled for each plane.
    """

    # Open the header file and load it as a json.
    with open(headerFileName, 'r') as headerFile:
        planesData = headerFile.read()
    planesList = json.loads(planesData)

    # Append a row of three 0s and a 1 to the constant transform.
    constantTransform = np.vstack((constantTransform,
                                   np.array([0, 0, 0, 1])))
    planePoints = [[] for _ in planesList]  # List to return.
    planeNames = []  # Instantiate a list to return.

    # Loop for all planes in the header file.
    for planeIndex, plane in enumerate(planesList):

        planeNames.append(plane["plane_name"])  # Record plane name.

        # Loop for all the points in each plane.
        s = sample(plane["robot_arm_data_json_file_names"],
                   sampleSize)
        for pointFileName in s:

            # Open the file for each point and load it as a json.
            with open(pointFileName, 'r') as pointFile:
                pointData = pointFile.read()
            pointJSON = json.loads(pointData)

            # Initialize and compute the transformation
            # from the base to the end effector.  Convert
            # the Euler angles to a rotation matrix.  Also
            # Changes angles to be -180 to 180 degrees
            # as Kinova uses this standard not 0-360 degrees.
            tEffector = np.zeros((3, 4))
            transAngles = pointJSON[0]["data"][:6]
            components = [float(v["value"].split(' ')[0])
                          for v in transAngles]
            angles = [(360-angle if angle < 0 else angle)
                      for angle in components[3:]]

            # scipy.spatial.transform.Rotation specifically.
            tEffector[0:3, 0:3] = Rotation.from_euler("xyz",
                                                      angles,
                                                      degrees=True
                                                     ).as_matrix()
            tEffector[0:3, 3] = components[:3]

            # Append a row of 3 0s and a 1 to the effector transform.
            tEffector = np.vstack((tEffector, np.array([0, 0, 0, 1])))

            # Calculate and record the point in the plane.
            point = (tEffector @ constantTransform)[0:3, 3]
            planePoints[planeIndex].append(point)

    # Return the tensor of 3D points and the planeNames.
    return np.array(planePoints), planeNames


def parsePoints2(fileNames: list[str],
                 sampleSize: int,
                 threshold: int,
                ) -> tuple[list[list[float]], list[str]]:
    """
    Takes in a list of filename specifying csv files that are properly
    formatted and returns a list of all the points for each plane,
    i.e., on for each item in the list fileNames.  The return is a 3D
    tensor which is number of planes by number of points by 3 since
    each coordinate has 3 components as they are in R3.  Ad hoc for
    this problem and the csvs provided.  If N Points are specified
    in the csvs, then sampleSize points are sampled for each plane.
    Removes any points with any components above threshold.
    """

    # Read the files into a pandas dataframe
    # and extract the x, y, and z coordinates.
    # Additionally, construct plane names.
    files = [pd.read_csv(fileName) for fileName in fileNames]
    xyzs = np.array([np.array([file["Tx"], file["Ty"], file["Tz"]]).T
                     for file in files])
    planeNames = [fileName[:fileName.index('.')]
                  for fileName in fileNames]

    planePoints = []  # A container to hold the planePoints.

    # Loop for all planes.
    for points in xyzs:

        plane = []  # A container to hold points in this plane.

        # Loop for all points in the plane.
        # s = sample(points, sampleSize)  # Sample the points.
        for xyz in points:

            # Check all components are within the threshold.
            if all([-threshold < c and c < threshold for c in xyz]):
                plane.append(list(xyz))  # Keep the point.

        planePoints.append(plane)  # Append the plane.

    # Return the tensor of 3D points and the planeNames.
    return planePoints, planeNames


def visualizeResults(planes: list[Plane],
                     intersections: np.ndarray,
                     terms: list[str],
                     axisLengths: list[float],
                     planePoints: np.ndarray,
                     graphPlanes: bool=True,
                     graphPoints: bool=True,
                     graphIntersections: bool=True
                    ) -> None:
    """
    Visualize the planes, ax + by + cz + d = 0, their
    intersection points, and the points that were used
    to find the fit planes in 3D to evaluate the results.
    3 boolean flags for what to graph are also taken as input.
    """

    data = []  # Initialize a container for the graphs.

    # Check if planes should be graphed.
    if graphPlanes:

        # Loop for all planes.
        for plane, term in zip(planes, terms):

            # Calculate and record the meshgrids.
            x, y, z = plane.getMeshGrids(term, axisLengths)
            data.append(go.Surface(x=x, y=y, z=z,
                                opacity=0.6,
                                showscale=False))

    # Check if intersections should be graphed.
    if graphIntersections:

        # Add the 3D intersection points to be graphed.
        if intersections.size > 0:
            data.append(go.Scatter3d(x=intersections[:, 0],
                                    y=intersections[:, 1],
                                    z=intersections[:, 2],
                                    mode="markers"))

    # Check if points should be graphed.
    if graphPoints:

        # Add the 3D points to be graphed.
        for point in planePoints:
            point = np.array(point)  # Cast to ndarray.
            data.append(go.Scatter3d(x=point[:, 0],
                                    y=point[:, 1],
                                    z=point[:, 2],
                                    mode="markers"))

    # Create and a layout
    # and show the figure.
    scene = {"xaxis": dict(title="x",
                           range=[axisLengths[0],
                           axisLengths[1]]),
             "yaxis": dict(title="y",
                           range=[axisLengths[2],
                           axisLengths[3]]),
             "zaxis": dict(title="z",
                           range=[axisLengths[4],
                           axisLengths[5]])}
    layout = go.Layout(width=1024, height=1024, scene=scene)

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def run(n: int = math.inf) -> None:
    """
    The main function used for all execution and to not pollute the
    global namespace. Parts (A) - (F) define all parts of the code.
    Takes in a variable to n for sampling points.  Uses all points
    if n = math.inf (default).  n != 0 Works for parsePoints1(),
    however it is NOT fully implemented for parsePoints2().
    """

    # (A) Variables needed as input.
    fileNames = ["top.csv",
                 "bottom.csv",
                 "front.csv",
                 "back.csv",
                 "left.csv",
                 "right.csv"]
    headerFileName = "header.json"
    constantTransform = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0.139]])

    ##################################################################
    # (B) Get the plane points, where planePoints is a 3D tensor
    # which is the number of planes by number of points by 3
    # since each coordinate has 3 components as they are in R3.
    # Either (1) generate planePoints or (2) read the points
    # from a file using the ad hoc function parsePoints().  The
    # "terms" variable controls if the plane is graphed in terms of
    # x, y, or z and th axes are only for vissualization purposes and
    # specify the x, y, and z min and max boundaries.

    # (1) Initialize an ideal box made from 6 planes, then generate
    # npoints points close to each plane within a close boundary
    # of [-b, b] from the plane with uniform noise n.
    # npoints, n, b = 4, 0.01, 10
    # box = [[0, 0, 1, 1], [0, 0, 1, -1],
    #        [0, 1, 0, 1], [0, 1, 0, -1],
    #        [1, 0, 0, 1], [1, 0, 0, -1]]
    # box = [Plane(*plane) for plane in box]  # Construct box.
    # planePoints = np.array([[plane.randomPoint(n, b)
    #                          for _ in range(npoints)]
    #                         for plane in box])
    # planeNames = ["top", "bottom", "right", "left", "front", "back"]
    # terms = ['z', 'z', 'y', 'y', 'x', 'x']  # Terms to graph in.
    # axisLength = (-150, 150, -100, 100, 100, -250)  # Axe lengths.

    # (2) Read the points and the names of the planes
    # from a file using one of the ad hoc parsePoint functions.

    # planePoints, planeNames = parsePoints1(headerFileName,
    #                                        constantTransform, n)
    planePoints, planeNames = parsePoints2(fileNames, n, 1000)

    terms = ['x', 'x', 'z', 'z', 'y', 'y']  # Terms to graph in.
    axisLength = (-150, 150, -100, 100, 100, -250)  # Axe lengths.

    ##################################################################
    # (C) Generate planes from the planePoints and
    # print all the equations of the estimated planes.
    planes = [Plane.estimatePlane(np.array(points))
              for points in planePoints.copy()]
    print("Estimated Planes:", *planes, sep='\n', end="\n\n")

    ##################################################################
    # (D) Calculate and print the intersection points of
    # any 3 planes any from a set of all the plans found.
    intersections = Plane.solveIntersection(planes)
    print("Intersection Points:",
          *intersections,
          sep='\n',
          end="\n\n")

    ##################################################################
    # (E) Check results by getting the angles between
    # the normal vectors of every pair of planes, then
    # print nicely.  Change flag to use radians or degrees.
    radians = False
    namesAngles = Plane.anglesBetweenPlanes(planes,
                                            planeNames,
                                            radians)

    angleStr = " radians" if radians else "\N{DEGREE SIGN}"
    formatStr = "The angle between \"{}\" and \"{}\" is {}{}"
    formatted = [formatStr.format(*field, angleStr)
                 for field in namesAngles]
    print("Angles between Planes:", *formatted, sep='\n', end='\n')

    ##################################################################
    # (F) Visualize the results.
    visualizeResults(planes,
                     intersections,
                     terms,
                     axisLength,
                     planePoints,
                     graphPlanes=True,
                     graphPoints=True,
                     graphIntersections=False)


if __name__ == "__main__":

    run()  # Execute script.

    # Run this to sample the points (5-9 in this case.  Change it if
    # you want) and see how accuracy changes based on sampling.  Works
    # for parsePoints1(). NOT fully implemented for parse points 2.
    # for n in range(5, 10):
    #     print(f"Sampling {n} Points:")
    #     l = []
    #     for _ in range(30):
    #         l.append(run(n))  # Execute script.
    #     l = np.array(l)
    #     print(f"Mean: {np.mean(l)}, STD: {np.std(l)}")
