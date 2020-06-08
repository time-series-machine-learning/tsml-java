package tsml.classifiers.distance_based.distances.dtw;


import experiments.data.DatasetLoading;
import java.math.BigDecimal;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.knn.KNN;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.distance_based.utils.instance.ExposedDenseInstance;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.InstanceTools;
import utilities.Utilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

/**
 * DTW distance measure.
 * <p>
 * Contributors: goastler
 */
public class DTWDistance extends BaseDistanceMeasure implements DTW {

    // the distance matrix produced by the distance function
    protected double[][] distanceMatrix;
    // whether to keep the distance matrix
    protected boolean keepDistanceMatrix = false;
    protected int warpingWindow = -1;
    protected double warpingWindowPercentage = -1;
    protected boolean warpingWindowInPercentage = false;

    public DTWDistance() {
    }

    public DTWDistance(int warpingWindow) {
        this();
        setWarpingWindow(warpingWindow);
    }

    public int getWarpingWindow() {
        return warpingWindow;
    }

    @Override
    public void setWarpingWindowPercentage(final double percentage) {
        this.warpingWindowPercentage = percentage;
        warpingWindowInPercentage = true;
    }

    @Override
    public double getWarpingWindowPercentage() {
        return warpingWindowPercentage;
    }

    public void setWarpingWindow(int warpingWindow) {
        this.warpingWindow = warpingWindow;
        warpingWindowInPercentage = false;
    }

    public double[][] getDistanceMatrix() {
        return distanceMatrix;
    }

    protected void setDistanceMatrix(double[][] distanceMatrix) {
        if(keepDistanceMatrix) {
            this.distanceMatrix = distanceMatrix;
        }
    }

    public boolean isKeepDistanceMatrix() {
        return keepDistanceMatrix;
    }

    public void setKeepDistanceMatrix(final boolean keepDistanceMatrix) {
        this.keepDistanceMatrix = keepDistanceMatrix;
    }

    @Override
    public void clean() {
        super.clean();
        cleanDistanceMatrix();
    }

    public void cleanDistanceMatrix() {
        distanceMatrix = null;
    }

    @Override
    public boolean isWarpingWindowInPercentage() {
        return warpingWindowInPercentage;
    }

    protected double squaredDifference(Instance a, int i, Instance b, int j) {
        return Math.pow(a.value(i) - b.value(j), 2);
    }

    protected double squaredDifference(double[] a, int i, double[] b, int j) {
        return Math.pow(a[i] - b[j], 2);
    }

    public double distance(double[] a, double[] b, final double limit) {

        double minDist;
        boolean tooBig;

        int aLength = a.length;
        int bLength = b.length;

        // put a or first as the longest time series
        if(bLength > aLength) {
            double[] tmp = a;
            a = b;
            b = tmp;
            aLength = a.length;
            bLength = b.length;
        }

        /*  Parameter 0<=r<=1. 0 == no warpingWindow, 1 == full warpingWindow
         generalised for variable window size
         * */
        int windowSize;
        if(warpingWindowInPercentage) {
            if(warpingWindowPercentage < 0) {
                windowSize = aLength;
            } else {
                windowSize = (int) (warpingWindowPercentage * aLength);
            }
        } else {
            if(warpingWindow < 0) {
                windowSize = aLength;
            } else {
                windowSize = warpingWindow;
            }
        }
        windowSize++; // to include current cell

        //Extra memory than required, could limit to windowsize,
        //        but avoids having to recreate during CV
        //for varying window sizes
        double[][] distances = new double[aLength][bLength];
        setDistanceMatrix(distances);

        /*
         //Set boundary elements to max.
         */
        int start, end;
        for(int i = 0; i < aLength; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = Math.min(i + windowSize + 1, bLength);
            for(int j = start; j < end; j++) {
                distances[i][j] = Double.POSITIVE_INFINITY;
            }
        }
        distances[0][0] = squaredDifference(a, 0, b, 0);
        //a is the longer series.
        //Base cases for warping 0 to all with max interval	r
        //Warp first[0] onto all second[1]...second[r+1]
        for(int j = 1; j < windowSize && j < bLength; j++) {
            distances[0][j] = distances[0][j - 1] + squaredDifference (a, 0, b, j);
        }

        //	Warp second[0] onto all first[1]...first[r+1]
        for(int i = 1; i < windowSize && i < aLength; i++) {
            distances[i][0] = distances[i - 1][0] + squaredDifference (a, i, b, 0);
        }
        //Warp the rest,
        for(int i = 1; i < aLength; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = Math.min(i + windowSize, bLength);
            if(distances[i][start - 1] < limit) {
                tooBig = false;
            }
            for(int j = start; j < end; j++) {
                minDist = Utilities.min(distances[i][j - 1], distances[i - 1][j], distances[i - 1][j - 1]);
                distances[i][j] =
                    minDist + squaredDifference(a, i, b, j);
                if(tooBig && distances[i][j] < limit) {
                    tooBig = false;
                }
            }
            //Early abandon
            if(tooBig) {
                return Double.POSITIVE_INFINITY;
            }
        }
        //Find the minimum distance at the end points, within the warping window.
        double distance = distances[aLength - 1][bLength - 1];
        return distance;
    }

    public double da(Instance ai, Instance bi, double limit) {

        double minDist;
        boolean tooBig;
        double[] a = ai.toDoubleArray();
        double[] b = bi.toDoubleArray();

        int aLength = a.length - 1;
        int bLength = b.length - 1;

        // put a or first as the longest time series
        if(bLength > aLength) {
            double[] tmp = a;
            a = b;
            b = tmp;
            aLength = a.length - 1;
            bLength = b.length - 1;
        }

        /*  Parameter 0<=r<=1. 0 == no warpingWindow, 1 == full warpingWindow
         generalised for variable window size
         * */
        int windowSize;
        if(warpingWindowInPercentage) {
            if(warpingWindowPercentage < 0) {
                windowSize = aLength;
            } else {
                windowSize = (int) (warpingWindowPercentage * aLength);
            }
        } else {
            if(warpingWindow < 0) {
                windowSize = aLength;
            } else {
                windowSize = warpingWindow;
            }
        }
        windowSize++; // to include current cell

        //Extra memory than required, could limit to windowsize,
        //        but avoids having to recreate during CV
        //for varying window sizes
        double[][] distances = new double[aLength][bLength];
        setDistanceMatrix(distances);

        /*
         //Set boundary elements to max.
         */
        int start, end;
        for(int i = 0; i < aLength; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = Math.min(i + windowSize + 1, bLength);
            for(int j = start; j < end; j++) {
                distances[i][j] = Double.POSITIVE_INFINITY;
            }
        }
        distances[0][0] = squaredDifference(a, 0, b, 0);
        //a is the longer series.
        //Base cases for warping 0 to all with max interval	r
        //Warp first[0] onto all second[1]...second[r+1]
        for(int j = 1; j < windowSize && j < bLength; j++) {
            distances[0][j] = distances[0][j - 1] + squaredDifference (a, 0, b, j);
        }

        //	Warp second[0] onto all first[1]...first[r+1]
        for(int i = 1; i < windowSize && i < aLength; i++) {
            distances[i][0] = distances[i - 1][0] + squaredDifference (a, i, b, 0);
        }
        //Warp the rest,
        for(int i = 1; i < aLength; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = Math.min(i + windowSize, bLength);
            if(distances[i][start - 1] < limit) {
                tooBig = false;
            }
            for(int j = start; j < end; j++) {
                minDist = Utilities.min(distances[i][j - 1], distances[i - 1][j], distances[i - 1][j - 1]);
                distances[i][j] =
                    minDist + squaredDifference(a, i, b, j);
                if(tooBig && distances[i][j] < limit) {
                    tooBig = false;
                }
            }
            //Early abandon
            if(tooBig) {
                return Double.POSITIVE_INFINITY;
            }
        }
        //Find the minimum distance at the end points, within the warping window.
        double distance = distances[aLength - 1][bLength - 1];
        return distance;
    }

    @Override
    public double distance(Instance ai, Instance bi, final double limit) {

        double minDist;
        boolean tooBig;
        double[] a = ExposedDenseInstance.extractAttributes(ai);
        double[] b = ExposedDenseInstance.extractAttributes(ai);

        int aLength = a.length - 1;
        int bLength = b.length - 1;

        // put a or first as the longest time series
        if(bLength > aLength) {
            double[] tmp = a;
            a = b;
            b = tmp;
            aLength = a.length - 1;
            bLength = b.length - 1;
        }

        /*  Parameter 0<=r<=1. 0 == no warpingWindow, 1 == full warpingWindow
         generalised for variable window size
         * */
        int windowSize;
        if(warpingWindowInPercentage) {
            if(warpingWindowPercentage < 0) {
                windowSize = aLength;
            } else {
                windowSize = (int) (warpingWindowPercentage * aLength);
            }
        } else {
            if(warpingWindow < 0) {
                windowSize = aLength;
            } else {
                windowSize = warpingWindow;
            }
        }
        windowSize++; // to include current cell

        //Extra memory than required, could limit to windowsize,
        //        but avoids having to recreate during CV
        //for varying window sizes
        double[][] distances = new double[aLength][bLength];
        setDistanceMatrix(distances);

        /*
         //Set boundary elements to max.
         */
        int start, end;
        for(int i = 0; i < aLength; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = Math.min(i + windowSize + 1, bLength);
            for(int j = start; j < end; j++) {
                distances[i][j] = Double.POSITIVE_INFINITY;
            }
        }
        distances[0][0] = squaredDifference(a, 0, b, 0);
        //a is the longer series.
        //Base cases for warping 0 to all with max interval	r
        //Warp first[0] onto all second[1]...second[r+1]
        for(int j = 1; j < windowSize && j < bLength; j++) {
            distances[0][j] = distances[0][j - 1] + squaredDifference (a, 0, b, j);
        }

        //	Warp second[0] onto all first[1]...first[r+1]
        for(int i = 1; i < windowSize && i < aLength; i++) {
            distances[i][0] = distances[i - 1][0] + squaredDifference (a, i, b, 0);
        }
        //Warp the rest,
        for(int i = 1; i < aLength; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = Math.min(i + windowSize, bLength);
            if(distances[i][start - 1] < limit) {
                tooBig = false;
            }
            for(int j = start; j < end; j++) {
                minDist = Utilities.min(distances[i][j - 1], distances[i - 1][j], distances[i - 1][j - 1]);
                distances[i][j] =
                    minDist + squaredDifference(a, i, b, j);
                if(tooBig && distances[i][j] < limit) {
                    tooBig = false;
                }
            }
            //Early abandon
            if(tooBig) {
                return Double.POSITIVE_INFINITY;
            }
        }
        //Find the minimum distance at the end points, within the warping window.
        double distance = distances[aLength - 1][bLength - 1];
        return distance;
    }

    @Override
    public ParamSet getParams() {
        return super.getParams().add(DTW.getWarpingWindowFlag(), getWarpingWindow());
    }

    @Override
    public void setParams(final ParamSet param) {
        ParamHandler.setParam(param, DTW.getWarpingWindowFlag(), this::setWarpingWindow, Integer.class);
    }

    public static double[] randomArray(final int length, final Random random, double min, double max) {
        final double[] array = new double[length];
        final double tmp = Math.max(max, min);
        min = Math.min(max, min);
        max = tmp;
        final double diff = max - min;
        for(int i = 0; i < length; i++) {
            array[i] = random.nextDouble() * diff + min;
        }
        return array;
    }


    private static class CustomInstance extends DenseInstance {

        public CustomInstance(final Instance instance) {
            super(instance);
        }

        @Override
        public double[] toDoubleArray() {
            return m_AttValues;
        }
    }

    public static long[] test(final int length, final Random random) {
        double[] a = randomArray(length, random, 0, 10);
        double[] b = randomArray(length, random, 0, 10);
        Instances i = InstanceTools.toWekaInstances(new double[][]{a, b}, new double[] {-1, -1});
        Instance ai = new CustomInstance(i.get(0));
        Instance bi = new CustomInstance(i.get(1));
        final DTWDistance df = new DTWDistance();
        df.setWarpingWindow(-1);
        long t1 = System.nanoTime();
        double d1 = df.distance(ai, bi, Double.POSITIVE_INFINITY, null);
        t1 = System.nanoTime() - t1;
        long t2 = System.nanoTime();
        double d2 = df.distance(a, b, Double.POSITIVE_INFINITY);
        t2 = System.nanoTime() - t2;
        long t3 = System.nanoTime();
        double d3 = df.da(ai, bi, Double.POSITIVE_INFINITY);
        t3 = System.nanoTime() - t3;
        Assert.assertEquals(d1, d2, 0d);
        Assert.assertEquals(d2, d3, 0f);
        return new long[] {t1, t2, t3};
    }

    public static Average[] multiTest(final int length, final Random random, final int count) {
        final Average a1 = new Average();
        final Average a2 = new Average();
        final Average a3 = new Average();
        for(int n = 0; n < count; n++) {
            final long[] times = test(length, random);
            a1.add(times[0]);
            a2.add(times[1]);
            a3.add(times[2]);
        }
        return new Average[] {a1, a2, a3};
    }

    private static class Average {
        private double firstValue = 0;
        private int n = 0;
        private BigDecimal sum = new BigDecimal(0);
        private BigDecimal sqSum = new BigDecimal(0);

        public void add(double v) {
            if(n == 0) {
                firstValue = v;
            }
            final double diff = v - firstValue;
            final double sqDiff = Math.pow(diff, 2);
            sum = sum.add(new BigDecimal(diff));
            sqSum = sqSum.add(new BigDecimal(sqDiff));
            n++;
        }

        public void remove(double v) {
            if(n <= 0) {
                throw new IllegalArgumentException();
            }
            n--;
            final double diff = v - firstValue;
            final double sqDiff = Math.pow(diff, 2);
            sum = sum.subtract(new BigDecimal(diff));
            sqSum = sqSum.subtract(new BigDecimal(sqDiff));
        }

        public double getMean() {
            return firstValue + sum.divide(new BigDecimal(n), BigDecimal.ROUND_HALF_UP).doubleValue();
        }

        public double getPopulationVariance() {
            if(n <= 0) {
                return 0;
            }
            return sqSum.subtract(sum.multiply(sum).divide(new BigDecimal(n), BigDecimal.ROUND_HALF_UP)).divide(new BigDecimal(n),
                BigDecimal.ROUND_HALF_UP).doubleValue();
        }

        public double getSampleVariance() {
            if(n <= 0) {
                return 0;
            }
            return sqSum.subtract(sum.multiply(sum).divide(new BigDecimal(n), BigDecimal.ROUND_HALF_UP)).divide(new BigDecimal(n - 1),
                BigDecimal.ROUND_HALF_UP).doubleValue();
        }

        public int getCount() {
            return n;
        }
    }

    public static void exp(int... lengths) {
        final int count = 30;
        for(int i = 0; i < lengths.length; i++) {
            final Random random = new Random(0);
            final Average[] avgs = multiTest(lengths[i], random, count);
            System.out.println(lengths[i] + ", " + avgs[0].getMean() + ", " + avgs[1].getMean() + ", " + avgs[2].getMean() + ", " + avgs[0].getPopulationVariance() + ", " + avgs[1].getPopulationVariance() + ", " + avgs[2].getPopulationVariance());
        }
    }

    public static void main(String[] args) throws Exception {
//        Thread.sleep(10000);
//        System.out.println("warm up");
//        exp(1000);
//        System.out.println();
//        System.out.println("the real thing");
//        exp(10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000);



//        final Average avg = new Average();
//        avg.add(1);
//        avg.add(2);
//        avg.add(3);
//        avg.add(4);
//        avg.add(5);
//        System.out.println(avg.getMean());
//        System.out.println(avg.getPopulationVariance());
//        System.out.println(avg.getCount());

        for(int i = 0; i < 10; i++) {
            long time;
            final Instances[] split = DatasetLoading.sampleBeef(0);
            KNNLOOCV knn = new KNNLOOCV();
            knn.setEstimateOwnPerformance(true);
            knn.setDistanceFunction(new DTWa(-1));
            knn.setSeed(0);
            time = System.nanoTime();
            knn.buildClassifier(split[0]);
            long t1 = System.nanoTime() - time;
            knn.setSeed(0);
            knn.setRebuild(true);
            knn.setDistanceFunction(new DTWDistance(-1));
            time = System.nanoTime();
            knn.buildClassifier(split[0]);
            long t2 = System.nanoTime() - time;
            System.out.println(t1);
            System.out.println(t2);
        }
    }
}
