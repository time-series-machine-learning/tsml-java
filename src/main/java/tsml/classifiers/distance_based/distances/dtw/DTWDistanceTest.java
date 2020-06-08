package tsml.classifiers.distance_based.distances.dtw;

import java.util.Random;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceTest;
import tsml.classifiers.distance_based.utils.instance.ExposedDenseInstance;
import utilities.InstanceTools;
import utilities.Utilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: test dtw
 * <p>
 * Contributors: goastler
 */
public class DTWDistanceTest {

    public static Instances buildInstances() {
        return InstanceTools.toWekaInstancesWithClass(new double[][] {
            {1,2,3,4,5,0},
            {6,11,15,2,7,1}
        });
    }

    private Instances instances;
    private DTWDistance df;

    @Before
    public void before() {
        instances = buildInstances();
        df = new DTWDistance();
        df.setInstances(instances);
    }

    @Test
    public void testFullWarp() {
        df.setWindowSize(-1);
        Assert.assertFalse(df.isWindowSizeInPercentage());
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 203, 0);
    }

    @Test
    public void testFullWarpPercentage() {
        df.setWindowSizePercentage(-1);
        Assert.assertTrue(df.isWindowSizeInPercentage());
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 203, 0);
    }

    @Test
    public void testConstrainedWarp() {
        df.setWindowSize(2);
        Assert.assertFalse(df.isWindowSizeInPercentage());
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 212, 0);
    }

    @Test
    public void testConstrainedWarpPercentage() {
        df.setWindowSizePercentage(0.5);
        Assert.assertTrue(df.isWindowSizeInPercentage());
        double distance = df.distance(instances.get(0), instances.get(1));
        Assert.assertEquals(distance, 212, 0);
    }

    @Test
    public void testRandomSetups() {
        final int min = -5;
        final int max = 5;
        final int length = 100;
        final int count = 10000;
        final double[] resultsOrig = new double[count];
        final double[] resultsCustom = new double[count];
        int limitCount = 0;
        for(int i = 0; i < count; i++) {
            final Random random = new Random(i);
            final double[] a = ERPDistanceTest.buildRandomArray(random, length, min, max);
            final double[] b = ERPDistanceTest.buildRandomArray(random, length, min, max);
            a[a.length - 1] = 1;
            b[b.length - 1] = 1;
            final Instances instances = InstanceTools.toWekaInstances(new double[][]{a, b}, new double[] {1,1});
            final Instance ai = instances.get(0);
            final Instance bi = instances.get(1);
            int window;
            if(random.nextBoolean()) {
                window = random.nextInt(length);
            } else {
                window = -1;
            }
            double limit = Double.POSITIVE_INFINITY;
            if(random.nextBoolean()) {
                limit = random.nextDouble() * 1000;
            }
            resultsOrig[i] = origDtw(ai, bi, limit, window);
            if(resultsOrig[i] == Double.POSITIVE_INFINITY) {
                limitCount++;
            }
            final DTWDistance df = new DTWDistance();
            df.setKeepMatrix(true);
            df.setWindowSize(window);
            resultsCustom[i] = df.distance(ai, bi, limit);
        }
        Assert.assertArrayEquals(resultsCustom,  resultsOrig, 0d);
    }

    private static double origDtw(Instance first, Instance second, double limit, int windowSize) {

        double minDist;
        boolean tooBig;

        int aLength = first.numAttributes() - 1;
        int bLength = second.numAttributes() - 1;

        /*  Parameter 0<=r<=1. 0 == no warpingWindow, 1 == full warpingWindow
         generalised for variable window size
         * */
//        int windowSize = warpingWindow + 1; // + 1 to include the current cell
//        if(warpingWindow < 0) {
//            windowSize = aLength + 1;
//        }
        if(windowSize < 0) {
            windowSize = first.numAttributes() - 1;
        } else {
            windowSize++;
        }
        //Extra memory than required, could limit to windowsize,
        //        but avoids having to recreate during CV
        //for varying window sizes
        double[][] distanceMatrix = new double[aLength][bLength];

        /*
         //Set boundary elements to max.
         */
        int start, end;
        for(int i = 0; i < aLength; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = Math.min(i + windowSize + 1, bLength);
            for(int j = start; j < end; j++) {
                distanceMatrix[i][j] = Double.POSITIVE_INFINITY;
            }
        }
        distanceMatrix[0][0] = (first.value(0) - second.value(0)) * (first.value(0) - second.value(0));
        //a is the longer series.
        //Base cases for warping 0 to all with max interval	r
        //Warp first[0] onto all second[1]...second[r+1]
        for(int j = 1; j < windowSize && j < bLength; j++) {
            distanceMatrix[0][j] =
                distanceMatrix[0][j - 1] + (first.value(0) - second.value(j)) * (first.value(0) - second.value(j));
        }

        //	Warp second[0] onto all first[1]...first[r+1]
        for(int i = 1; i < windowSize && i < aLength; i++) {
            distanceMatrix[i][0] =
                distanceMatrix[i - 1][0] + (first.value(i) - second.value(0)) * (first.value(i) - second.value(0));
        }
        //Warp the rest,
        for(int i = 1; i < aLength; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = Math.min(i + windowSize, bLength);
            if(distanceMatrix[i][start - 1] < limit) {
                tooBig = false;
            }
            for(int j = start; j < end; j++) {
                minDist = distanceMatrix[i][j - 1];
                if(distanceMatrix[i - 1][j] < minDist) {
                    minDist = distanceMatrix[i - 1][j];
                }
                if(distanceMatrix[i - 1][j - 1] < minDist) {
                    minDist = distanceMatrix[i - 1][j - 1];
                }
                distanceMatrix[i][j] =
                    minDist + (first.value(i) - second.value(j)) * (first.value(i) - second.value(j));
                if(tooBig && distanceMatrix[i][j] < limit) {
                    tooBig = false;
                }
            }
            //Early abandon
            if(tooBig) {
                return Double.POSITIVE_INFINITY;
            }
        }
        //Find the minimum distance at the end points, within the warping window.
        double distance = distanceMatrix[aLength - 1][bLength - 1];
        return distance;



//        double[] a = ExposedDenseInstance.extractAttributeValuesAndClassLabel(first);
//        double[] b = ExposedDenseInstance.extractAttributeValuesAndClassLabel(bi);
//        double[][] matrixD = null;
//        double minDist;
//        boolean tooBig;
//        // Set the longest series to a. is this necessary?
//        double[] temp;
//        if(a.length<b.length){
//            temp=a;
//            a=b;
//            b=temp;
//        }
//        int n=a.length-1;
//        int m=b.length-1;
///*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp
//generalised for variable window size
//* */
////        windowSize = getWindowSize(n);
//        //Extra memory than required, could limit to windowsize,
//        //        but avoids having to recreate during CV
//        //for varying window sizes
//        if(matrixD==null)
//            matrixD=new double[n][m];

/*
//Set boundary elements to max.
*/
//        int start,end;
//        for(int i=0;i<n;i++){
//            start=windowSize<i?i-windowSize:0;
//            end=i+windowSize+1<m?i+windowSize+1:m;
//            for(int j=start;j<end;j++)
//                matrixD[i][j]=Double.MAX_VALUE;
//        }
//        matrixD[0][0]=(a[0]-b[0])*(a[0]-b[0]);
//        //a is the longer series.
//        //Base cases for warping 0 to all with max interval	r
//        //Warp a[0] onto all b[1]...b[r+1]
//        for(int j=1;j<windowSize && j<m;j++)
//            matrixD[0][j]=matrixD[0][j-1]+(a[0]-b[j])*(a[0]-b[j]);
//
//        //	Warp b[0] onto all a[1]...a[r+1]
//        for(int i=1;i<windowSize && i<n;i++)
//            matrixD[i][0]=matrixD[i-1][0]+(a[i]-b[0])*(a[i]-b[0]);
//        //Warp the rest,
//        for (int i=1;i<n;i++){
//            tooBig=true;
//            start=windowSize<i?i-windowSize+1:1;
//            end=i+windowSize<m?i+windowSize:m;
//            for (int j = start;j<end;j++){
//                minDist=matrixD[i][j-1];
//                if(matrixD[i-1][j]<minDist)
//                    minDist=matrixD[i-1][j];
//                if(matrixD[i-1][j-1]<minDist)
//                    minDist=matrixD[i-1][j-1];
//                matrixD[i][j]=minDist+(a[i]-b[j])*(a[i]-b[j]);
//                if(tooBig&&matrixD[i][j]<cutoff)
//                    tooBig=false;
//            }
//            //Early abandon
//            if(tooBig){
//                return Double.MAX_VALUE;
//            }
//        }
//        //Find the minimum distance at the end points, within the warping window.
//        return matrixD[n-1][m-1];
    }
}
