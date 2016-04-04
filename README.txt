Group Assignment 2: Shape Google
Sanmi Oyenuga (ojo), Clemence Lapeyre (acl26), Gray Williams

Link to Github Repository: https://github.com/grayman82/Group2_ShapeGoogle

_____________________________________________________________________________
Time Spent on Assignment: ~2 weeks (used 3 late days) Most of the time was
                          spent optimizing! The functions were initially 
                          written with for-loops and checked. The output was 
                          correct, but we did not realize that we were doing 
                          the work of the numpy histogram function, updating 
                          bins in hist and creating the interval arrays that 
                          we then fed to the plotting functions directly.... 
                          Then we ran into problems when generating the recall 
                          graphs because of the slow for-loops. Every function 
                          had to be restructured for broadcasting.
                          Technically, we wrote the assignment three times in
                          the prescribed period + 3 extra days ;)
Assigment Feedback: 
Assignment Concerns: No concerns.
_____________________________________________________________________________


#############################################################################
samplePointCloud

This function alters the point cloud of an image to make it invariant to rigid
3D transformations (translation and scale). N points were sampled randomly 
from the mesh, generating the point cloud Ps. The point cloud was first 
centered on its centroid to account for translation and then scaled by a factor
's' to account for variation in scale. The scaling factor was calculated to 
ensure an RMS distance to the origin of 1. Broadcasting techniques were used.
#############################################################################

#############################################################################
getShapeHistogram

The function generates a histogram which shows the number of points within a 
certain radius interval from the center of the point cloud. This was initially 
implemented by first centering the point cloud and then looping through all 
the points in the point cloud to calculate the distance of each point from 
the origin. The result of integer division (distance//interval) gave the bin 
index of the histogram that the point belonged to. This array at that bin 
index was then incremented by 1. This was later changed to make use of numpy's
histogram function. For efficiency, the for-loop was also removed and 
calculations were done with broadcasting. 
#############################################################################

#############################################################################
getShapeShellHistogram

This function is an extension of getShapeHistogram; it first categorizes the
points based on radial distance from the center and then again by direction.
The directions were uniformly distributed along the surface of the sphere.
To keep data rotation invariant, the histogram displays the number of points
in decreasing sorted order for the sectors of a particular shell. The
function makes use of broadcasting and the numpy function for creating 2D
histograms.
#############################################################################

#############################################################################
getShapeHistogramPCA

GRAY
#############################################################################

#############################################################################
getD2Histogram

This function randomly generates NSamples pointpairs, calculates the Euclidean 
distance between each pair, and then generates a histogram. Broadcasting was 
used to avoid very costly for-loops. Originally, for-loops had been used to
identify bad point pairs (i.e., same point), but this slowed down the recall
graph generation significantly. This could have been fixed by generating
the point pairs using a random permutation and the mod function.
#############################################################################

#############################################################################
getA3Histogram

This function randomly generates NSamples point-triples, calculates the 
interior angle between them, and generates a histogram. Vectors were created
from the points and then dot products, norms, and arccos were used to find
the angle. Replacing the initial for-loop with broadcasting was necessary
to generate the recall graphs. However, the for-loop ensured that no triple
generated an invalid result (2+ points being the same would create a 0 vector
that could result in division by zero). The numpy function nan_to_num was
used on the final array of angles to replace the the NaN values with zero.
Alternatively, the point triples could have been generated using a random 
permutation and the mod function. 
#############################################################################

#############################################################################
getEGIHistogram

GRAY
#############################################################################

#############################################################################
getSpinImage

This function makes a spin image! After the point cloud has been aligned with
its PCA axes, it is rotated around the axis of greatest variation (here the 
x-axis). A histogram of the projection of the point cloud onto the other two 
axes is generated for a range of angles. The data from these individual 
histograms is summed to create the data for the final image. This was 
difficult to optimize with broadcasting and no for-loop. In fact, the final 
implementation still contains a for-loop for the calculation and summation of
the histograms, which did not seem possible to broadcast. Performing the 
rotation outside the loop did significantly reduce the time for creating the 
recall graphs. Rotation with quaternions may have further improved this.
#############################################################################
____________________________________________________________________________

Note: The functions below could have been implemented with broadcasting, but
      optimiation resources were scarce
____________________________________________________________________________

#############################################################################
compareHistsEuclidean

This function computes the euclidean distance between a set of histograms. It
returns a matrix D where D_ij is the distance between the histogram for 
point cloud i and point cloud j.
#############################################################################

#############################################################################
compareHistsCosine

This function computes the cosine distance between a set of histograms. It
returns a matrix D where D_ij is (1 - cosdistance) between the histogram for 
point cloud i and point cloud j. 
#############################################################################

#############################################################################
compareHistsChiSquared

This function computes the chi squared distance between a set of histograms. 
It returns a matrix D where D_ij is the distance between the histogram for 
point cloud i and point cloud j. It should be noted that this metric cannot
be used on histograms with zero values.
#############################################################################

#############################################################################
compareHistsEMD1D

This function computes the 1D Earth mover's distance between a set of 
histograms. The numpy function cumsum was used for the CDF. It returns a 
matrix D where D_ij is the distance between the histogram for point cloud i 
and point cloud j. This metric works best for 1D histograms. 
#############################################################################

#############################################################################
Performance Evaluation/ Precision Recall Graphs

This was implemented by first looping through all rows in the similarity matrix.
For every row,the class for the row number was calculated by integer division
(current row// point clouds per class) and the row shapes were sorted in
increasing order. We then loop through the elements of each sorted row while
while skipping shapes queried against themselves. For each entry, if the class
of the entry is equal to the class off the row, the precision was calculated as
the number of correct shapes encountered/total number of shapes encountered, and
the result precision array was generated.

-------

For the graph results we observe that the precision curves on these graphs that
are closest to the value precision (y) = 1 i.e. whatever curve maximizes the
area under the curve were better i.e. offered better statistics.
This is because higher precision values translate to better discernment between
shapes in the histograms and similarity matrix. With a better way of separating
shapes, we are presented with less uncertainty in picking our shapes (better
statistics)


Recall Graph Results:

6Descriptors.png / 4Descriptors.png:
Create precision recall graphs comparing every descriptor you implemented with
reasonable parameter choices for each one.

getShapeHist.png:
Create precision recall graphs which show the effect of choosing different
numbers of bins for the basic shell histogram

getD2Hist.png:
Create precision recall graphs which show the effect of choosing different
numbers of random samples for the D2 histogram

getD2DistanceMetrics.png / EGIDistanceMetrics.png:
Create at least two precision recall graphs using different metrics on the
same descriptor and showing the effect. 

EGI.png:
At least one other precision recall graph;how does the number of sphere sampled
normal directions affect the results for the extended Gaussian Image.
#############################################################################

#############################################################################
Classification Contest

TBD
#############################################################################


