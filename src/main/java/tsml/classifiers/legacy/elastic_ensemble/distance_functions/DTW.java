/*
DTW with early abandon
 */
package tsml.classifiers.legacy.elastic_ensemble.distance_functions;

import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.WarpingPathResults;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.GenericTools;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author ajb
 */
public final class DTW extends DTW_DistanceBasic {


    /**
     * @param a
     * @param b
     * @param cutoff
     * @return
     */
    @Override
    public final double distance(double[] a, double[] b, double cutoff) {
        double minDist;
        boolean tooBig;
// Set the longest series to a. is this necessary?
        double[] temp;
        if (a.length < b.length) {
            temp = a;
            a = b;
            b = temp;
        }
        int n = a.length;
        int m = b.length;
/*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp 
generalised for variable window size
* */
        windowSize = getWindowSize(n);
//Extra memory than required, could limit to windowsize,
//        but avoids having to recreate during CV 
//for varying window sizes        
        if (matrixD == null)
            matrixD = new double[n][m];
        
/*
//Set boundary elements to max. 
*/
        int start, end;
        for (int i = 0; i < n; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = i + windowSize + 1 < m ? i + windowSize + 1 : m;
            for (int j = start; j < end; j++)
                matrixD[i][j] = Double.MAX_VALUE;
        }
        matrixD[0][0] = (a[0] - b[0]) * (a[0] - b[0]);
//a is the longer series. 
//Base cases for warping 0 to all with max interval	r	
//Warp a[0] onto all b[1]...b[r+1]
        for (int j = 1; j < windowSize && j < m; j++)
            matrixD[0][j] = matrixD[0][j - 1] + (a[0] - b[j]) * (a[0] - b[j]);

//	Warp b[0] onto all a[1]...a[r+1]
        for (int i = 1; i < windowSize && i < n; i++)
            matrixD[i][0] = matrixD[i - 1][0] + (a[i] - b[0]) * (a[i] - b[0]);
//Warp the rest,
        for (int i = 1; i < n; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = i + windowSize < m ? i + windowSize : m;
            for (int j = start; j < end; j++) {
                minDist = matrixD[i][j - 1];
                if (matrixD[i - 1][j] < minDist)
                    minDist = matrixD[i - 1][j];
                if (matrixD[i - 1][j - 1] < minDist)
                    minDist = matrixD[i - 1][j - 1];
                matrixD[i][j] = minDist + (a[i] - b[j]) * (a[i] - b[j]);
                if (tooBig && matrixD[i][j] < cutoff)
                    tooBig = false;
            }
            //Early abandon
            if (tooBig) {
                return Double.MAX_VALUE;
            }
        }
//Find the minimum distance at the end points, within the warping window. 
        return matrixD[n - 1][m - 1];
    }


    /************************************************************************************************
     Support for FastEE
     ************************************************************************************************/
    private final static int MAX_SEQ_LENGTH = 4000;
    private final static int DIAGONAL = 0;                  // value for diagonal
    private final static int LEFT = 1;                      // value for left
    private final static int UP = 2;
    private final static double[][] distMatrix = new double[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH];
    private final static int[][] minDistanceToDiagonal = new int[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH];

    public static WarpingPathResults distanceExt(final Instance first, final Instance second, final int windowSize) {
        if(first.attribute(0).isRelationValued())
            return distanceExtMultivariate( first, second, windowSize);
        double minDist = 0.0;
        final int n = first.numAttributes() - 1;
        final int m = second.numAttributes() - 1;

        double diff;
        int i, j, indiceRes, absIJ;
        int jStart, jEnd, indexInfyLeft;

        diff = first.value(0) - second.value(0);
        distMatrix[0][0] = diff * diff;
        minDistanceToDiagonal[0][0] = 0;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = first.value(i) - second.value(0);
            distMatrix[i][0] = distMatrix[i - 1][0] + diff * diff;
            minDistanceToDiagonal[i][0] = i;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {
            diff = first.value(0) - second.value(j);
            distMatrix[0][j] = distMatrix[0][j - 1] + diff * diff;
            minDistanceToDiagonal[0][j] = j;
        }
        if (j < m) distMatrix[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + windowSize + 1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0) distMatrix[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                absIJ = Math.abs(i - j);
                indiceRes = GenericTools.argMin3(distMatrix[i - 1][j - 1], distMatrix[i][j - 1], distMatrix[i - 1][j]);
                switch (indiceRes) {
                    case DIAGONAL:
                        minDist = distMatrix[i - 1][j - 1];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j - 1]);
                        break;
                    case LEFT:
                        minDist = distMatrix[i][j - 1];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i][j - 1]);
                        break;
                    case UP:
                        minDist = distMatrix[i - 1][j];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j]);
                        break;
                }
                diff = first.value(i) - second.value(j);
                distMatrix[i][j] = minDist + diff * diff;
            }
            if (j < m) distMatrix[i][j] = Double.POSITIVE_INFINITY;
        }

        WarpingPathResults resExt = new WarpingPathResults();
        resExt.distance = distMatrix[n - 1][m - 1];
        resExt.distanceFromDiagonal = minDistanceToDiagonal[n - 1][m - 1];
        return resExt;
    }

    private static double multivariatePointDistance(Instances data1, Instances data2, int posA, int posB){
        double diff=0;
        for(int i=0;i<data1.numInstances();i++)
            diff += (data1.instance(i).value(posA) - data2.instance(i).value(posB))*(data1.instance(i).value(posA) - data2.instance(i).value(posB));
        return diff;
    }
    public static WarpingPathResults distanceExtMultivariate(final Instance first, final Instance second, final int windowSize) {
        Instances data1=first.relationalValue(0);
        Instances data2=second.relationalValue(0);
        double minDist = 0.0;
        final int n = first.numAttributes() - 1;
        final int m = second.numAttributes() - 1;

        double diff=0;
        int i, j, indiceRes, absIJ;
        int jStart, jEnd, indexInfyLeft;
        diff=multivariatePointDistance(data1,data2,0,0);
        distMatrix[0][0] = diff;
        minDistanceToDiagonal[0][0] = 0;
        for (i = 1; i < Math.min(n, 1 + windowSize); i++) {
            diff = multivariatePointDistance(data1,data2,i,0);
            distMatrix[i][0] = distMatrix[i - 1][0] + diff;
            minDistanceToDiagonal[i][0] = i;
        }

        for (j = 1; j < Math.min(m, 1 + windowSize); j++) {

            diff =multivariatePointDistance(data1,data2,0,j);
            distMatrix[0][j] = distMatrix[0][j - 1] + diff;
            minDistanceToDiagonal[0][j] = j;
        }
        if (j < m) distMatrix[0][j] = Double.POSITIVE_INFINITY;

        for (i = 1; i < n; i++) {
            jStart = Math.max(1, i - windowSize);
            jEnd = Math.min(m, i + windowSize + 1);
            indexInfyLeft = i - windowSize - 1;
            if (indexInfyLeft >= 0) distMatrix[i][indexInfyLeft] = Double.POSITIVE_INFINITY;

            for (j = jStart; j < jEnd; j++) {
                absIJ = Math.abs(i - j);
                indiceRes = GenericTools.argMin3(distMatrix[i - 1][j - 1], distMatrix[i][j - 1], distMatrix[i - 1][j]);
                switch (indiceRes) {
                    case DIAGONAL:
                        minDist = distMatrix[i - 1][j - 1];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j - 1]);
                        break;
                    case LEFT:
                        minDist = distMatrix[i][j - 1];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i][j - 1]);
                        break;
                    case UP:
                        minDist = distMatrix[i - 1][j];
                        minDistanceToDiagonal[i][j] = Math.max(absIJ, minDistanceToDiagonal[i - 1][j]);
                        break;
                }
                diff = multivariatePointDistance(data1,data2,i,j);
                distMatrix[i][j] = minDist + diff;
            }
            if (j < m) distMatrix[i][j] = Double.POSITIVE_INFINITY;
        }

        WarpingPathResults resExt = new WarpingPathResults();
        resExt.distance = distMatrix[n - 1][m - 1];
        resExt.distanceFromDiagonal = minDistanceToDiagonal[n - 1][m - 1];
        return resExt;
    }



    public static int getWindowSize(final int n, final double r) {
        return (int) (r * n);
    }

}
