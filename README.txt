Group Assignment 2


Sanmi Oyenuga (ojo), Clemence Lapeyre, Gray Williams

#############################################################################

GetShapeHistogram

- This was initially implemented by first centering the point cloud and then
looping through all the points in the point cloud to calculate the distance of
each point from the origin. the result of integer division (distance//interval)
gave what bin of the histogram the point in question fell into at this bin is
incremented. However this implementation was optimized by eliminating the for
loop and calculating the distances in a broadcast function. The histogram was
then generated using the numpy histogram function.

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
 shapes, we are presented with less uncertainty in picking our shapes(better
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







     EGI.png
