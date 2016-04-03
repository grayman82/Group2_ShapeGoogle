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

Recall Graph Results 


In this graph by far 100,000 samples is better than 100!  The best precision
 recall graphs are the ones with the highest precision values across the whole
  graph.  So the ones that are closest to the line precision(recall) = 1,
  or alternatively the ones with the most area.  Graphs with higher precision
   values correspond to statistics which are more discerning; that is, they
    correctly separate out shapes in the histogram space better, so that
     there are fewer confusions/mix-ups
