#Purpose: To implement a suite of 3D shape statistics and to use them for point
#cloud classification
#TODO: Fill in all of this code for group assignment 2
import sys
sys.path.append("S3DGLPy")
from Primitives3D import *
from PolyMesh import *

import numpy as np
import matplotlib.pyplot as plt
import math

POINTCLOUD_CLASSES = ['biplane', 'desk_chair', 'dining_chair', 'fighter_jet', 'fish', 'flying_bird', 'guitar', 'handgun', 'head', 'helicopter', 'human', 'human_arms_out', 'potted_plant', 'race_car', 'sedan', 'shelves', 'ship', 'sword', 'table', 'vase']

NUM_PER_CLASS = 10

#########################################################
##                UTILITY FUNCTIONS                    ##
#########################################################

#Purpose: Export a sampled point cloud into the JS interactive point cloud viewer
#Inputs: Ps (3 x N array of points), Ns (3 x N array of estimated normals),
#filename: Output filename
def exportPointCloud(Ps, Ns, filename):
    N = Ps.shape[1]
    fout = open(filename, "w")
    fmtstr = "%g" + " %g"*5 + "\n"
    for i in range(N):
        fields = np.zeros(6)
        fields[0:3] = Ps[:, i]
        fields[3:] = Ns[:, i]
        fout.write(fmtstr%tuple(fields.flatten().tolist()))
    fout.close()

#Purpose: To sample a point cloud, center it on its centroid, and
#then scale all of the points so that the RMS distance to the origin is 1
def samplePointCloud(mesh, N):
    (Ps, Ns) = mesh.randomlySamplePoints(N)
    # accounting for translation-- center the point cloud on its centroid
    centroid = np.mean(Ps,1)[:, None] # 3x1 matrix with the mean of each row
    Ps_centered = Ps - centroid # center the point cloud
    # accounting for scale-- RMS distance of each point to the origin is 1
    Ps_c_squared = Ps_centered**2 # squares each element of the points in the point cloud
    col_sum = np.sum(Ps_c_squared, 0) # sum across the columns
    row_sum = np.sum(col_sum) # sum across the row
    s = (float(N)/row_sum)**0.5 # plug in calculated values and solve for s
    Ps_new = Ps_centered*s # normalize by 's'
    return (Ps_new, Ns)

#Purpose: To sample the unit sphere as evenly as possible.  The higher
#res is, the more samples are taken on the sphere (in an exponential
#relationship with res).  By default, samples 66 points
def getSphereSamples(res = 2):
    m = getSphereMesh(1, res)
    return m.VPos.T

#Purpose: To compute PCA on a point cloud
#Inputs: X (3 x N array representing a point cloud)
#Returns: (eigs, V) where eigs are the eigenvalues sorted in decreasing order and
#V is a 3x3 matrix with each row being the eigenvector corresponding to the eigenvalue
#of the same index in eigs
def doPCA(X):
    if len(X) == 0:
        return (np.zeros(3), np.eye(3))
    A = np.dot(X, X.T)
    (eigs, V) = np.linalg.eig(A) #retrieves eigenvalues and column matrix of eigenvectors
    eig_tuples = zip(eigs, V.T)
    eig_tuples = sorted(eig_tuples, key= lambda eig_pair: eig_pair[0], reverse=True) #sorting eigs in decreasing order
    (eigs, V) = zip(*eig_tuples)
    return (eigs, V)

#########################################################
##                SHAPE DESCRIPTORS                    ##
#########################################################

#Purpose: To compute a shape histogram, counting points distributed in concentric spherical shells centered at the origin
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency)
#NShells (number of shells), RMax (maximum radius)
#Returns: hist (histogram of length NShells)
def getShapeHistogram(Ps, Ns, NShells, RMax):
    hist = np.zeros(NShells) #initialize histogram values to zero
    centroid = np.mean(Ps,1)[:, None] #find centroid of point cloud
    Ps_centered = Ps - centroid #center point cloud at origin
    distances = np.linalg.norm(Ps_centered, axis = 0) #calculate point distance from origin
    hist, bins = np.histogram(distances, bins = int(NShells), range=[0, float(RMax)]) #generate histogram
    return hist 

#Purpose: To create shape histogram with concentric spherical shells and
#sectors within each shell, sorted in decreasing order of number of points
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NShells (number of shells),
#RMax (maximum radius), SPoints: A 3 x S array of points sampled evenly on
#the unit sphere (get these with the function "getSphereSamples")
def getShapeShellHistogram(Ps, Ns, NShells, RMax, SPoints):
    NSectors = SPoints.shape[1] #A number of sectors equal to the number of points sampled on the sphere
    hist = np.zeros((NShells, NSectors)) #Create a 2D histogram that is NShells x NSectors & initialize with zeros
    centroid = np.mean(Ps,1)[:, None]
    Ps_centered = Ps - centroid
    distances = np.linalg.norm(Ps_centered, axis = 0) #calculate point distance from origin
    shells = distances//(float(RMax)/NShells) #calculate shell values
    dots = np.dot(Ps_centered.T, SPoints) #calculate dot products
    sectors = np.argmax(dots, axis = 1) #calculate sector values
    #generate histogram
    hist, xedges, yedges = np.histogram2d(shells, sectors, bins=[int(NShells), int(NSectors)], range = [[0.0, float(NShells)],[0.0, float(NSectors)]]) 
    hist = np.fliplr(np.sort(hist)) # reverse-sort sectors in each shell
    return hist.flatten()

#Purpose: To create shape histogram with concentric spherical shells and to
#compute the PCA eigenvalues in each shell
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NShells (number of shells),
#RMax (maximum radius), sphereRes: An integer specifying points on thes phere
#to be used to cluster shells
def getShapeHistogramPCA(Ps, Ns, NShells, RMax):
    #Create a 2D histogram, with 3 eigenvalues for each shell
    hist = np.zeros((NShells, 3))
    centroid = np.mean(Ps,1)[:, None]
    Ps_centered = Ps - centroid
    Ps_organized = []
    interval = float(RMax)/NShells
    for i in range(NShells):
        Ps_organized.append([])
    for point in Ps_centered.T:
        tempDist = np.linalg.norm(point)
        pos = int(tempDist//interval)
        Ps_organized[pos].append(point)
    for i in range(NShells):
        (eigs, V) = doPCA(np.transpose(Ps_organized[i]))
        hist[i] = eigs
    return hist

#Purpose: To create shape histogram of the pairwise Euclidean distances between
#randomly sampled points in the point cloud
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), DMax (Maximum distance to consider),
#NBins (number of histogram bins), NSamples (number of pairs of points sample
#to compute distances)
def getD2Histogram(Ps, Ns, DMax, NBins, NSamples):
    hist = np.zeros(NBins)
    sampledPairs = np.random.randint(len(Ps[0]), size = (NSamples, 2.)) # get random point pairs
    P1A = Ps[:, sampledPairs[:,0]]
    P2A = Ps[:, sampledPairs[:,1]]
    TEMP = np.subtract(P1A, P2A)
    DISTANCES = np.linalg.norm(TEMP, axis =0)
    hist, bins= np.histogram(DISTANCES, bins = int(NBins), range=[0, float(DMax)])
    #plt.bar(bins, histogram, width= DMax / NBins * 0.9)
    #plt.show()
    return hist 


#Purpose: To create shape histogram of the angles between randomly sampled triples of points
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NBins (number of histogram bins),
#NSamples (number of triples of points sample to compute angles)
def getA3Histogram(Ps, Ns, NBins, NSamples):
    hist = np.zeros(NBins)
    sampledTriples = np.random.randint(len(Ps[0]), size= (NSamples, 3.)) # get random point triples
    P1A = Ps[:, sampledTriples[:,0]]
    P2A = Ps[:, sampledTriples[:,1]]
    P3A = Ps[:, sampledTriples[:,2]]
    U = np.subtract(P1A, P2A)
    V = np.subtract(P3A, P2A)
    UNORM = np.linalg.norm(U, axis = 0)
    VNORM = np.linalg.norm(V, axis = 0)
    UDOT = np.sum(U*V, axis = 0)
    UVNORM = np.multiply(UNORM, VNORM)
    ANGLES = np.arccos(np.divide(UDOT, UVNORM))
    ANGLES = np.nan_to_num(ANGLES)
    hist, bins= np.histogram(ANGLES, bins = int(NBins), range=[0, float(np.pi)])
    #plt.bar(bins, histogram, width= np.pi / NBins * 0.9)
    #plt.show()
    return hist 

#Purpose: To create the Extended Gaussian Image by binning normals to
#sphere directions after rotating the point cloud to align with its principal axes
#Inputs: Ps (3 x N point cloud) (use to compute PCA), Ns (3 x N array of normals),
#SPoints: A 3 x S array of points sampled evenly on the unit sphere used to
#bin the normals
def getEGIHistogram(Ps, Ns, SPoints):
    S = SPoints.shape[1]
    hist = np.zeros(S)
    (eigs, V) = doPCA(Ps)
    Ns_aligned = np.dot(V, Ns)
    dots2 = np.dot(Ns_aligned.T, SPoints)
    pos = np.argmax(dots2, axis = 1)
    hist, bins = np.histogram(pos, bins = int(S), range=[0, float(S)]) #generate histogram
    return hist

#Purpose: To create an image which stores the amalgamation of rotating
#a bunch of planes around the largest principal axis of a point cloud and
#projecting the points on the minor axes onto the image.
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals, not needed here),
#NAngles: The number of angles between 0 and 2*pi through which to rotate
#the plane, Extent: The extent of each axis, Dim: The number of pixels along
#each minor axis
def getSpinImage(Ps, Ns, NAngles, Extent, Dim):
    hist = np.zeros((Dim, Dim)) # create a 2D histogram for an image
    (eigs, V) = doPCA(Ps) # eigVals in decreasing order w. corresponding eigVecs in V
    Ps_aligned = np.dot(V, Ps) # project point cloud onto PCA axes
    # rotate the point cloud around axis of greatest variation (x-axis)
    angles_of_rotation = np.linspace(0, 360, NAngles+1)
    for i in range(len(angles_of_rotation)-1):
        theta = np.radians(angles_of_rotation[i])
        vals = np.array([1.0, 0.0, 0.0, 0.0, np.cos(theta), np.sin(-theta), 0.0, np.sin(theta), np.cos(theta)])
        R = np.reshape(vals, (3, 3)) # rotation matrix
        Ps_rotated = np.dot(R, Ps_aligned)
        # Bin the point cloud projected onto the other two axes
        H, xedges, yedges = np.histogram2d(Ps_rotated[1,:], Ps_rotated[2,:], bins=Dim, range = [[-Extent, Extent],[-Extent, Extent]]) 
        hist = hist + H # sum images
    fig1 = plt.figure()
    plt.pcolormesh(xedges, yedges, hist)
    plt.show() # display spin image
    return hist.flatten()

#Purpose: To create a histogram of spherical harmonic magnitudes in concentric
#spheres after rasterizing the point cloud to a voxel grid
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals, not used here),
#VoxelRes: The number of voxels along each axis (for instance, if 30, then rasterize
#to 30x30x30 voxels), Extent: The number of units along each axis (if 2, then
#rasterize in the box [-1, 1] x [-1, 1] x [-1, 1]), NHarmonics: The number of spherical
#harmonics, NSpheres, the number of concentric spheres to take
def getSphericalHarmonicMagnitudes(Ps, Ns, VoxelRes, Extent, NHarmonics, NSpheres):
    hist = np.zeros((NSpheres, NHarmonics))
    #TODO: Finish this
    return hist.flatten()

#Purpose: Utility function for wrapping around the statistics functions.
#Inputs: PointClouds (a python list of N point clouds), Normals (a python
#list of the N corresponding normals), histFunction (a function
#handle for one of the above functions), *args (addditional arguments
#that the descriptor function needs)
#Returns: AllHists (A KxN matrix of all descriptors, where K is the length
#of each descriptor)
def makeAllHistograms(PointClouds, Normals, histFunction, *args):
    N = len(PointClouds)
    #Call on first mesh to figure out the dimensions of the histogram
    h0 = histFunction(PointClouds[0], Normals[0], *args)
    K = h0.size
    AllHists = np.zeros((K, N))
    AllHists[:, 0] = h0
    for i in range(1, N):
        print "Computing histogram %i of %i..."%(i+1, N)
        AllHists[:, i] = histFunction(PointClouds[i], Normals[i], *args)
    return AllHists

#########################################################
##              HISTOGRAM COMPARISONS                  ##
#########################################################

#Purpose: helper method to normalize histograms by mass
#Inputs: hist, a 1D array of length K with the values of the histogram
# h'[i] = h[i] / sum (from k = 1 to K) h[k]
def normalizeHist(hist):
    sumHist = np.sum(hist) # sum (from k=1 to K) h[k] #fixed
    hist_prime = hist / float(sumHist) # use broadcasting; h'[i] = h[i] / sumHist
    return hist_prime

#Purpose: To compute the euclidean distance between a set
#of histograms
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the Euclidean
#distance between the histogram for point cloud i and point cloud j)
def compareHistsEuclidean(AllHists):
    N = AllHists.shape[1] # number of columns aka number of point clouds / histograms
    D = np.zeros((N, N))
    for i in range (N): 
        pc1 = normalizeHist(AllHists[:, i]) # normalize histogram i
        for j in range (N): 
            pc2 = normalizeHist(AllHists[:, j]) # normalize histogram j
            # dist = sqrt ( (pc1_1 - pc2_1)^2 + ... + (pc1_K - pc2_K)^2 )
            D[i][j] = np.linalg.norm(np.subtract(pc1, pc2))
    return D

#Purpose: To compute the cosine distance between a set
#of histograms
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the cosine
#distance between the histogram for point cloud i and point cloud j)
def compareHistsCosine(AllHists):
    N = AllHists.shape[1]  # number of columns aka number of point clouds / histograms
    D = np.zeros((N, N))
    for i in range (N): # could change this to range (N-1) for efficiency?
        pc1 = normalizeHist(AllHists[:, i]) # normalize histogram i
        for j in range (N): # could change this to range (i+1, N) for efficiency?
            pc2 = normalizeHist(AllHists[:, j]) # normalize histogram j
            # dist = (v_i dot v_j) / (|v_i|*|v_j|)
            numerator = np.dot(pc1, pc2) # v_i dot v_j
            denominator = np.linalg.norm(pc1)*np.linalg.norm(pc2)
            dist = numerator / denominator
            D[i][j] = 1-dist # assign cosine distance value for ij
    return D

#Purpose: To compute the cosine distance between a set
#of histograms
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the chi squared
#distance between the histogram for point cloud i and point cloud j)
def compareHistsChiSquared(AllHists):
    # note that this can only be used on histograms with non-zero values
    N = AllHists.shape[1]  # number of columns aka number of point clouds / histograms
    D = np.zeros((N, N))
    for i in range (N):
        pc1 = normalizeHist(AllHists[:, i]) # normalize histogram i
        for j in range (N): 
            pc2 = normalizeHist(AllHists[:, j]) # normalize histogram j
            # dist = 0.5*{(sum from k=1 to K) [ (h1[k]-h2[k])^2 / (h1[k] + h2[k]) ]}
            numerator = (np.subtract(pc1, pc2))**2 # element-wise subtraction, element-wise square
            denominator = np.add(pc1, pc2) # element-wise addition
            summation = np.sum(numerator/denominator) # element-wise division, sum over array
            D[i][j] = 0.5*summation # scale & assign distance value for ij
    return D


#Purpose: To compute the 1D Earth mover's distance between a set
#of histograms (note that this only makes sense for 1D histograms)
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the earth mover's
#distance between the histogram for point cloud i and point cloud j)
def compareHistsEMD1D(AllHists):
    N = AllHists.shape[1]  # number of columns aka number of point clouds / histograms
    D = np.zeros((N, N))
    for i in range (N): # 
        pc1 = normalizeHist(AllHists[:, i]) # normalize histogram i
        for j in range (N):
            pc2 = normalizeHist(AllHists[:, j]) # normalize histogram j
            # dist = (sum from k = 1 to K) | hC_i[k] - hC_j[k]|
            hci= np.cumsum(pc1)
            hcj= np.cumsum(pc2)
            emd = np.sum(np.absolute(np.subtract(hci,hcj)))
            D[i][j] = emd 
    return D


#########################################################
##              CLASSIFICATION CONTEST                 ##
#########################################################

#Purpose: To implement your own custom distance matrix between all point
#clouds for the point cloud clasification contest
#Inputs: PointClouds, an array of point cloud matrices, Normals: an array
#of normal matrices
#Returns: D: A N x N matrix of distances between point clouds based
#on your metric, where Dij is the distnace between point cloud i and point cloud j
def getMyShapeDistances(PointClouds, Normals):
    #TODO: Finish this
    #This is just an example, but you should experiment to find which features
    #work the best, and possibly come up with a weighted combination of
    #different features
    HistsD2 = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 100000)
    DEuc = compareHistsEuclidean(HistsD2)
    return DEuc

#########################################################
##                     EVALUATION                      ##
#########################################################

#Purpose: To return an average precision recall graph for a collection of
#shapes given the similarity scores of all pairs of histograms.
#Inputs: D (An N x N matrix, where the ij entry is the earth mover's distance
#between the histogram for point cloud i and point cloud j).  It is assumed
#that the point clouds are presented in contiguous chunks of classes, and that
#there are "NPerClass" point clouds per each class (for the dataset provided
#there are 10 per class so that's the default argument).  So the program should
#return a precision recall graph that has 9 elements
#Returns PR, an (NPerClass-1) length array of average precision values for all
#recalls
def getPrecisionRecall(D, NPerClass = 10):
    PR = np.zeros(NPerClass-1) #initialize precision value arrays with zeros
    rIn = 0 #initialize count index for number of rows
    for row in D: #for every row in the similarity matrix
        classval = rIn//NPerClass #find the class of current row. i.e since increments of NPerClass belong to same class
        #integer division should floor all values in same class to same class value e.g. 30,31,32...39 become 3
        sortRow = np.argsort(row) #sort row in question and return the indexes of values
        count = 1 #initialize count for total number of shapes looked at for now
        correct = 1 #initialize count for number of shapes in correct class looked at for now
        for entry in sortRow: #sort through every element in sorted row
            if (rIn == entry): #If shape is being queried against itself
                #count+= 1 #increment number ofshapes looked at
                continue #then skip this iteration
            if (entry//NPerClass == classval): #if the class of the current entry is equal to the class of querying entry do this
                precision = correct/count #calculate precision i.e. fraction of  shapes in the correct class over the fraction of shapes looked at
                PR[correct - 1] += precision #add precision value of shape in (correct - 1) index to the rest of the precision values in that index
                correct += 1 #increment my counter for shapes in correct class
            count+= 1 #increment counter for all shapes looked at.
            if(correct >= NPerClass-1): #if we've found all correct shapes in class, no need to proceed, break
                break
        rIn += 1 #increment row counter i.e. move to next row
    PR = PR/len(D) #divide all precision values by number of rows i.e find average as summation of precision values/number of precision values
    return PR


#########################################################
##                     MAIN TESTS                      ##
#########################################################

#sys.exit("Not executing rest of code")


if __name__ == '__main__':
    #NRandSamples = 10000 #You can tweak this number
    #np.random.seed(100) #For repeatable results randomly sampling
    #Load in and sample all meshes
    #PointClouds = []
    #Normals = []
    #for i in range(len(POINTCLOUD_CLASSES)):
    #for i in range(1):
        #print "LOADING CLASS %i of %i..."%(i, len(POINTCLOUD_CLASSES))
        #PCClass = []
        #for j in range(NUM_PER_CLASS):
        #for j in range(3):
            #m = PolyMesh()
            #filename = "models_off/%s%i.off"%(POINTCLOUD_CLASSES[i], j)
            #print "Loading ", filename 
            #m.loadOffFileExternal(filename)
            #(Ps, Ns) = samplePointCloud(m, NRandSamples)
            #PointClouds.append(Ps)
            #Normals.append(Ns)
    #Ps0 = PointClouds[0]
    #Ps1 = PointClouds[1] 
    #Ps2 = PointClouds[2] 
    #DMax = 3 
    #NBins = 4
    #Samples = 100
    #ist0 = getD2Histogram(Ps0, Normals[0], DMax, NBins, NSamples) 
    #print hist0
    #hist1 = getD2Histogram(Ps1, Normals[1], DMax, NBins, NSamples) 
    #print hist1
    #hist2 = getD2Histogram(Ps2, Normals[0], DMax, NBins, NSamples) 
    #print hist2
    #AllHists = np.zeros((4,3))
    #AllHists [:,0] = hist0
    #AllHists [:,1] = hist1
    #AllHists [:,2] = hist2
    #D = compareHistsEMD1D(AllHists)
    #print D
   #m = PolyMesh()
   #m.loadFile("models_off/biplane0.off") #Load a mesh
   #(Ps, Ns) = samplePointCloud(m, 10000) #Sample 20,000 points and associated normals
   #exportPointCloud(Ps, Ns, "biplane.pts") #Export point cloud

   #TESTING GET-SHAPE-HISTOGRAM
   #histogram1 = getShapeHistogram(Ps, Ns, 21, 3)
   #plt.bar(bins1, histogram1, width=3.0/21*0.9)
   #plt.show()

   #TESTING GET-2D-HISTOGRAM
   #DMax = 4
   #NBins = 20
   #NSamples = 5
   #histogram =  getD2Histogram(Ps, Ns, DMax, NBins, NSamples)

   #TESTING GET-A3-HISTOGRAM
   #NBins = 2
   #NSamples = 10
   #histogram =  getA3Histogram(Ps, Ns, NBins, NSamples)
   #plt.bar(bins, histogram, width= math.pi / NBins * 0.9)
   #plt.show()

   #TESTING GET-SHAPE-SHELL-HISTOGRAM
   #NShells = 4
   #RMax = 2
   #SPoints = getSphereSamples() # res is auto-set to 2 (66 sample points)
   #histogram = getShapeShellHistogram(Ps, Ns, NShells, RMax, SPoints)
   #plt.bar(bins, histogram, width =  float(RMax)/NShells/SPoints.shape[1]*0.9)
   #plt.show()
   
   #TESTING GET-SPIN-IMAGE
   #NAngles = 720
   #Extent = 2
   #Dim = 1000
   #histogram = getSpinImage(Ps, Ns, NAngles, Extent, Dim)
   
#SPoints = getSphereSamples(2)
#HistsSpin = makeAllHistograms(PointClouds, Normals, getSpinImage, 100, 2, 40)
#HistsEGI = makeAllHistograms(PointClouds, Normals, getEGIHistogram, SPoints)
#HistsA3 = makeAllHistograms(PointClouds, Normals, getA3Histogram, 30, 100000)
#HistsD2 = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 100000)

#DSpin = compareHistsEuclidean(HistsSpin)
#DEGI = compareHistsEuclidean(HistsEGI)
#DA3 = compareHistsEuclidean(HistsA3)
#DD2 = compareHistsEuclidean(HistsD2)

#PRSpin = getPrecisionRecall(DSpin)
#PREGI = getPrecisionRecall(DEGI)
#PRA3 = getPrecisionRecall(DA3)
#PRD2 = getPrecisionRecall(DD2)

#recalls = np.linspace(1.0/9.0, 1.0, 9)
#plt.plot(recalls, PREGI, 'c', label='EGI')
#plt.hold(True)
##plt.plot(recalls, PRA3, 'k', label='A3')
#plt.plot(recalls, PRD2, 'r', label='D2')
#plt.plot(recalls, PRSpin, 'b', label='Spin')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.legend()
#plt.show()

    #TODO: Finish this, run experiments.  Also in the above code, you might
    #just want to load one point cloud and test your histograms on that first
    #so you don't have to wait for all point clouds to load when making
    #minor tweaks
