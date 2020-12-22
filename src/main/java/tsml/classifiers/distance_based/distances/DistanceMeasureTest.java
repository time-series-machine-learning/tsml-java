package tsml.classifiers.distance_based.distances;

import experiments.data.DatasetLoading;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTWDistanceConfigs;
import tsml.classifiers.distance_based.distances.ed.EDistanceConfigs;
import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceConfigs;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistanceConfigs;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistanceTest;
import tsml.classifiers.distance_based.distances.msm.MSMDistance;
import tsml.classifiers.distance_based.distances.msm.MSMDistanceConfigs;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.twed.TWEDistance;
import tsml.classifiers.distance_based.distances.twed.TWEDistanceConfigs;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceConfigs;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceTest;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearch;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import tsml.transformers.Derivative;
import utilities.FileUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import static experiments.data.DatasetLoading.*;

@RunWith(Parameterized.class)
public class DistanceMeasureTest {

    private enum DistanceMeasureSpaceBuilder implements ParamSpaceBuilder {
        DTW(new DTWDistanceConfigs.DTWSpaceBuilder()),
        DDTW(new DTWDistanceConfigs.DDTWSpaceBuilder()),
        WDTW(new WDTWDistanceConfigs.WDTWSpaceBuilder()),
        WDDTW(new WDTWDistanceConfigs.WDDTWSpaceBuilder()),
        ERP(new ERPDistanceConfigs.ERPSpaceBuilder()),
        LCSS(new LCSSDistanceConfigs.LCSSSpaceBuilder()),
        MSM(new MSMDistanceConfigs.MSMSpaceBuilder()),
        TWED(new TWEDistanceConfigs.TWEDSpaceBuilder()),
        ED(new EDistanceConfigs.EDSpaceBuilder()),
        ;
        

        DistanceMeasureSpaceBuilder(final ParamSpaceBuilder builder) {
            this.builder = builder;
        }

        private final ParamSpaceBuilder builder;

        @Override public ParamSpace build(final TimeSeriesInstances data) {
            return builder.build(data);
        }

        @Override public ParamSpace build(final Instances data) {
            return builder.build(data);
        }
    }
    
    @Parameterized.Parameters(name = "{0} {1}")
    public static Collection<Object[]> data() throws Exception {
        List<Object[]> space = new ArrayList<>();
        final String[] datasetNames = new String[] {
//                "ItalyPowerDemand"
//                , 
                "GunPoint"
//                , "BasicMotions"
//                , 
//                "Beef"
        };
        for(DistanceMeasureSpaceBuilder builder : DistanceMeasureSpaceBuilder.values()) {
            for(String datasetName : datasetNames) {
                space.add(new Object[] {builder, datasetName});
            }
        }
        return space;
    }

    @Parameterized.Parameter(1)
    public String datasetName;
    
    @Parameterized.Parameter(0)
    public DistanceMeasureSpaceBuilder paramSpaceBuilder;
    
    private static final Map<String, TimeSeriesInstances> dataLookup = new ConcurrentHashMap<>();
    
    @Test
    public void test() throws Exception {
        // conduct several random tests of random distance measure parameters + random insts
        final long time = System.nanoTime();
        final String testDistancesFilePath = "src/main/java/" + getClass().getPackage().getName().replaceAll("\\.", "/") + "/test/" + paramSpaceBuilder + "_" + datasetName + ".csv";
        final boolean generateDistances = !new File(testDistancesFilePath).exists();
        final int numDistances;
        final double[] distances;
        if(generateDistances) {
            numDistances = 1000;
            distances = new double[numDistances];
        } else {
            String content = FileUtils.readFromFile(testDistancesFilePath);
            String[] parts = content.split(",");
            distances = Arrays.stream(parts).mapToDouble(Double::parseDouble).toArray();
            numDistances = distances.length;
        }
        final String path;
        if(Arrays.asList(BAKED_IN_MTSC_DATASETS).contains(datasetName)) {
            path = BAKED_IN_MTSC_DATA_PATH;
        } else {
            path = BAKED_IN_TSC_DATA_PATH;
        }
        final Instances[] insts = DatasetLoading.sampleDataset(path, datasetName, 0);
        insts[0].addAll(insts[1]);
        final TimeSeriesInstances data = Converter.fromArff(insts[0]);
        final Random random = new Random(0);
        final RandomSearch randomSearch = new RandomSearch();
        randomSearch.setRandom(random);
        randomSearch.setIterationLimit(numDistances);
        final ParamSpace paramSpace = paramSpaceBuilder.build(data);
        randomSearch.buildSearch(paramSpace);
        for(int i = 0; i < numDistances; i++) {
            System.out.println(i);
            final int i1 = random.nextInt(data.numInstances());
            int i2 = 0;
            do {
                i2 = random.nextInt(data.numInstances());
            } while(i2 == i1);
            if(!randomSearch.hasNext()) {
                throw new IllegalStateException("random search should never run out");
            }
            final ParamSet paramSet = randomSearch.next();
            TimeSeriesInstance inst1 = data.get(i1);
            TimeSeriesInstance inst2 = data.get(i2);
            final DistanceMeasure distanceMeasure = paramSet.getSingle(DistanceMeasure.DISTANCE_MEASURE_FLAG);
            if(distanceMeasure instanceof MatrixBasedDistanceMeasure) {
                ((MatrixBasedDistanceMeasure) distanceMeasure).setGenerateDistanceMatrix(false);
            }
            final double distanceWithoutLimit = distanceMeasure.distance(inst1, inst2);
            if(generateDistances) {
                distances[i] = distanceWithoutLimit;
            } else {
                Assert.assertTrue(distanceWithoutLimit >= 0);
                Assert.assertEquals(distances[i], distanceWithoutLimit, 0d);
                final double limit = distanceWithoutLimit / 2;
                final double distanceWithLimit = distanceMeasure.distance(inst1, inst2, limit);
                Assert.assertTrue(
                        distanceWithLimit == distanceWithoutLimit || distanceWithLimit == Double.POSITIVE_INFINITY);
                if(distanceMeasure instanceof MatrixBasedDistanceMeasure) {
                    ((MatrixBasedDistanceMeasure) distanceMeasure).setGenerateDistanceMatrix(true);
                    final double distanceWithMatrix = distanceMeasure.distance(inst1, inst2);
                    Assert.assertEquals(distanceWithoutLimit, distanceWithMatrix, 0d);
                }
                double v = Double.NaN;
                double u = Double.NaN;
                boolean check = true;
                if(BAKED_IN_TSC_DATA_PATH.equals(path)) {
                    final int len = inst1.getMaxLength();
                    double window = 0;
                    switch(paramSpaceBuilder) {
                        case ERP:
                            window = ((ERPDistance) distanceMeasure).getWindowSize();
                            break;
                        case DTW:
                            window = ((DTWDistance) distanceMeasure).getWindowSize();
                            break;
                        case DDTW:
                            window = ((DTWDistance) ((TransformDistanceMeasure) distanceMeasure).getDistanceMeasure()).getWindowSize();
                            break;
                    }
                    final int rawWindow = (int) Math.ceil(window * len);
                    final double otherWindow = Math.min(1, Math.max(0, ((double) rawWindow) / len - 10e-12));
                    switch(paramSpaceBuilder) {
                        case LCSS:
                            u = distanceWithoutLimit;
                            v = LCSSDistanceTest.origLcss(Converter.toArff(inst1), Converter.toArff(inst2),
                                    Double.POSITIVE_INFINITY,
                                    (int) Math.ceil(((LCSSDistance) distanceMeasure).getWindowSize() * inst1.getMaxLength()),
                                    ((LCSSDistance) distanceMeasure).getEpsilon());
                            break;
                        case DDTW:
                            v = origDtw(Converter.toArff(new Derivative().transform(inst1)), Converter.toArff(new Derivative().transform(inst2)),
                                    Double.POSITIVE_INFINITY,
                                    rawWindow);
                            ((DTWDistance) ((TransformDistanceMeasure) distanceMeasure).getDistanceMeasure()).setWindowSize(otherWindow);
                            u = distanceMeasure.distance(inst1, inst2);
                            break;
                        case WDDTW:
                            v = origWdtw(Converter.toArff(new Derivative().transform(inst1)), Converter.toArff(new Derivative().transform(inst2)),
                                    Double.POSITIVE_INFINITY,
                                    ((WDTWDistance) ((TransformDistanceMeasure) distanceMeasure).getDistanceMeasure()).getG());
                            u = distanceMeasure.distance(inst1, inst2);
                            break;
                        case DTW:
                            v = origDtw(Converter.toArff(inst1), Converter.toArff(inst2),
                                    Double.POSITIVE_INFINITY,
                                    rawWindow);
                            ((DTWDistance) distanceMeasure).setWindowSize(otherWindow);
                            u = distanceMeasure.distance(inst1, inst2);
                            break;
                        case WDTW:
                            v = origWdtw(Converter.toArff(inst1), Converter.toArff(inst2),
                                    Double.POSITIVE_INFINITY,
                                    ((WDTWDistance) distanceMeasure).getG());
                            u = distanceMeasure.distance(inst1, inst2);
                            break;
                        case ERP:
                            v = origErp(Converter.toArff(inst1), Converter.toArff(inst2),
                                    Double.POSITIVE_INFINITY,
                                    rawWindow, ((ERPDistance) distanceMeasure).getG());
                            ((ERPDistance) distanceMeasure).setWindowSize(otherWindow);
                            u = distanceMeasure.distance(inst1, inst2);
                            break;
                        case MSM:
                            v = origMsm(Converter.toArff(inst1), Converter.toArff(inst2),
                                    Double.POSITIVE_INFINITY,
                                    ((MSMDistance) distanceMeasure).getC());
                            u = distanceMeasure.distance(inst1, inst2);
                            break;
                        case TWED:
                            v = origTwed(Converter.toArff(inst1), Converter.toArff(inst2),
                                    Double.POSITIVE_INFINITY,
                                    ((TWEDistance) distanceMeasure).getLambda(),((TWEDistance) distanceMeasure).getNu());
                            u = distanceMeasure.distance(inst1, inst2);
                            break;
                        default:
                            check = false;
                    }
                    if(check) Assert.assertEquals(u, v, 0d);
                }
            }
        }
        if(generateDistances) {
            FileUtils.writeToFile(StrUtils.join(",", distances), testDistancesFilePath);
            Assert.fail("distances computed in " + TimeUnit.SECONDS.convert(System.nanoTime() - time, TimeUnit.NANOSECONDS) + "s and written to " + testDistancesFilePath);
        }
    }


    public static double origErp(Instance first, Instance second, double limit, int band, double penalty) {

        int aLength = first.numAttributes() - 1;
        int bLength = second.numAttributes() - 1;

        // Current and previous columns of the matrix
        double[] curr = new double[bLength];
        double[] prev = new double[bLength];

        // size of edit distance band
        // bandsize is the maximum allowed distance to the diagonal
        //        int band = (int) Math.ceil(v2.getDimensionality() * bandSize);
        if(band < 0) {
            band = aLength + 1;
        }

        // g parameters for local usage
        double gValue = penalty;

        for(int i = 0;
                i < aLength;
                i++) {
            // Swap current and prev arrays. We'll just overwrite the new curr.
            {
                double[] temp = prev;
                prev = curr;
                curr = temp;
            }
            int l = i - (band + 1);
            if(l < 0) {
                l = 0;
            }
            int r = i + (band + 1);
            if(r > (bLength - 1)) {
                r = (bLength - 1);
            }

            boolean tooBig = true;

            for(int j = l;
                    j <= r;
                    j++) {
                if(Math.abs(i - j) <= band) {
                    // compute squared distance of feature vectors
                    double val1 = first.value(i);
                    double val2 = gValue;
                    double diff = (val1 - val2);
                    final double dist1 = diff * diff;

                    val1 = gValue;
                    val2 = second.value(j);
                    diff = (val1 - val2);
                    final double dist2 = diff * diff;

                    val1 = first.value(i);
                    val2 = second.value(j);
                    diff = (val1 - val2);
                    final double dist12 = diff * diff;

                    final double cost;

                    if((i + j) != 0) {
                        if((i == 0) || ((j != 0) && (((prev[j - 1] + dist12) > (curr[j - 1] + dist2)) && (
                                (curr[j - 1] + dist2) < (prev[j] + dist1))))) {
                            // del
                            cost = curr[j - 1] + dist2;
                        } else if((j == 0) || ((i != 0) && (((prev[j - 1] + dist12) > (prev[j] + dist1)) && (
                                (prev[j] + dist1) < (curr[j - 1] + dist2))))) {
                            // ins
                            cost = prev[j] + dist1;
                        } else {
                            // match
                            cost = prev[j - 1] + dist12;
                        }
                    } else {
                        cost = 0;
                    }

                    curr[j] = cost;

                    if(tooBig && cost < limit) {
                        tooBig = false;
                    }
                } else {
                    curr[j] = Double.POSITIVE_INFINITY; // outside band
                }
            }
            if(tooBig) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return curr[bLength - 1];
    }

    private static double[] generateWeights(int seriesLength, double g) {
        double halfLength = (double) seriesLength / 2;
        double[] weightVector = new double[seriesLength];
        for (int i = 0; i < seriesLength; i++) {
            weightVector[i] = 1d / (1d + Math.exp(-g * (i - halfLength)));
        }
        return weightVector;
    }

    public static double origWdtw(Instance a, Instance b, double limit, double g) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        double[] weightVector = generateWeights(aLength, g);

        //create empty array
        double[][] distances = new double[aLength][bLength];

        //first value
        distances[0][0] = (a.value(0) - b.value(0)) * (a.value(0) - b.value(0)) * weightVector[0];

        //early abandon if first values is larger than cut off
        if (distances[0][0] > limit) {
            return Double.POSITIVE_INFINITY;
        }

        //top row
        for (int i = 1; i < bLength; i++) {
            distances[0][i] =
                    distances[0][i - 1] + (a.value(0) - b.value(i)) * (a.value(0) - b.value(i)) * weightVector[i]; //edited by Jay
        }

        //first column
        for (int i = 1; i < aLength; i++) {
            distances[i][0] =
                    distances[i - 1][0] + (a.value(i) - b.value(0)) * (a.value(i) - b.value(0)) * weightVector[i]; //edited by Jay
        }

        //warp rest
        double minDistance;
        for (int i = 1; i < aLength; i++) {
            boolean overflow = true;

            for (int j = 1; j < bLength; j++) {
                //calculate distance_measures
                minDistance = Math.min(distances[i][j - 1], Math.min(distances[i - 1][j], distances[i - 1][j - 1]));
                distances[i][j] =
                        minDistance + (a.value(i) - b.value(j)) * (a.value(i) - b.value(j)) * weightVector[Math.abs(i - j)];

                if (overflow && distances[i][j] <= limit) {
                    overflow = false; // because there's evidence that the path can continue
                }
            }

            //early abandon
            if (overflow) {
                return Double.POSITIVE_INFINITY;
            }
        }
        return distances[aLength - 1][bLength - 1];
    }


    public static double origDtw(Instance first, Instance second, double limit, int windowSize) {

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


    private static double findCost(double newPoint, double x, double y, double c) {
        double dist = 0;

        if(((x <= newPoint) && (newPoint <= y)) ||
                   ((y <= newPoint) && (newPoint <= x))) {
            dist = c;
        } else {
            dist = c + Math.min(Math.abs(newPoint - x), Math.abs(newPoint - y));
        }

        return dist;
    }

    public static double origMsm(Instance a, Instance b, double limit, double c) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;

        double[][] cost = new double[aLength][bLength];

        // Initialization
        cost[0][0] = Math.abs(a.value(0) - b.value(0));
        for(int i = 1; i < aLength; i++) {
            cost[i][0] = cost[i - 1][0] + findCost(a.value(i), a.value(i - 1), b.value(0), c);
        }
        for(int i = 1; i < bLength; i++) {
            cost[0][i] = cost[0][i - 1] + findCost(b.value(i), a.value(0), b.value(i - 1), c);
        }

        // Main Loop
        double min;
        for(int i = 1; i < aLength; i++) {
            min = Double.POSITIVE_INFINITY;
            for(int j = 1; j < bLength; j++) {
                double d1, d2, d3;
                d1 = cost[i - 1][j - 1] + Math.abs(a.value(i) - b.value(j));
                d2 = cost[i - 1][j] + findCost(a.value(i), a.value(i - 1), b.value(j), c);
                d3 = cost[i][j - 1] + findCost(b.value(j), a.value(i), b.value(j - 1), c);
                cost[i][j] = Math.min(d1, Math.min(d2, d3));

            }
            for(int j = 0; j < bLength; j++) {
                min = Math.min(min, cost[i][j]);
            }
            if(min > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }
        // Output
        return cost[aLength - 1][bLength - 1];
    }


    public static double origTwed(Instance a, Instance b, double limit, double lambda, double nu) {

        int aLength = a.numAttributes() - 1;
        int bLength = b.numAttributes() - 1;
        int dim = 1;
        double dist, disti1, distj1;
        double[][] ta = new double[aLength][dim];
        double[][] tb = new double[bLength][dim];
        double[] tsa = new double[aLength];
        double[] tsb = new double[bLength];
        for(int i = 0; i < tsa.length; i++) {
            tsa[i] = (i + 1);
        }
        for(int i = 0; i < tsb.length; i++) {
            tsb[i] = (i + 1);
        }

        int r = ta.length;
        int c = tb.length;
        int i, j, k;
        //Copy over values
        for(i = 0; i < aLength; i++) {
            ta[i][0] = a.value(i);
        }
        for(i = 0; i < bLength; i++) {
            tb[i][0] = b.value(i);
        }

        /* allocations in c
	double **D = (double **)calloc(r+1, sizeof(double*));
	double *Di1 = (double *)calloc(r+1, sizeof(double));
	double *Dj1 = (double *)calloc(c+1, sizeof(double));
	for(i=0; i<=r; i++) {
		D[i]=(double *)calloc(c+1, sizeof(double));
	}
*/
        double[][] D = new double[r + 1][c + 1];
        double[] Di1 = new double[r + 1];
        double[] Dj1 = new double[c + 1];
        // local costs initializations
        for(j = 1; j <= c; j++) {
            distj1 = 0;
            for(k = 0; k < dim; k++) {
                if(j > 1) {
                    //CHANGE AJB 8/1/16: Only use power of 2 for speed up,
                    distj1 += (tb[j - 2][k] - tb[j - 1][k]) * (tb[j - 2][k] - tb[j - 1][k]);
                    // OLD VERSION                    distj1+=Math.pow(Math.abs(tb[j-2][k]-tb[j-1][k]),degree);
                    // in c:               distj1+=pow(fabs(tb[j-2][k]-tb[j-1][k]),degree);
                } else {
                    distj1 += tb[j - 1][k] * tb[j - 1][k];
                }
            }
            //OLD              		distj1+=Math.pow(Math.abs(tb[j-1][k]),degree);
            Dj1[j] = (distj1);
        }

        for(i = 1; i <= r; i++) {
            disti1 = 0;
            for(k = 0; k < dim; k++) {
                if(i > 1) {
                    disti1 += (ta[i - 2][k] - ta[i - 1][k]) * (ta[i - 2][k] - ta[i - 1][k]);
                }
                // OLD                 disti1+=Math.pow(Math.abs(ta[i-2][k]-ta[i-1][k]),degree);
                else {
                    disti1 += (ta[i - 1][k]) * (ta[i - 1][k]);
                }
            }
            //OLD                  disti1+=Math.pow(Math.abs(ta[i-1][k]),degree);

            Di1[i] = (disti1);

            for(j = 1; j <= c; j++) {
                dist = 0;
                for(k = 0; k < dim; k++) {
                    dist += (ta[i - 1][k] - tb[j - 1][k]) * (ta[i - 1][k] - tb[j - 1][k]);
                    //                  dist+=Math.pow(Math.abs(ta[i-1][k]-tb[j-1][k]),degree);
                    if(i > 1 && j > 1) {
                        dist += (ta[i - 2][k] - tb[j - 2][k]) * (ta[i - 2][k] - tb[j - 2][k]);
                    }
                    //                    dist+=Math.pow(Math.abs(ta[i-2][k]-tb[j-2][k]),degree);
                }
                D[i][j] = (dist);
            }
        }// for i

        // border of the cost matrix initialization
        D[0][0] = 0;
        for(i = 1; i <= r; i++) {
            //            D[i][0] = Double.POSITIVE_INFINITY;
            D[i][0] = D[i - 1][0] + Di1[i];
        }
        for(j = 1; j <= c; j++) {
            //            D[0][j] = Double.POSITIVE_INFINITY;
            D[0][j] = D[0][j - 1] + Dj1[j];
        }

        double dmin, htrans, dist0;
        int iback;

        for(i = 1; i <= r; i++) {
            for(j = 1; j <= c; j++) {
                htrans = Math.abs((tsa[i - 1] - tsb[j - 1]));
                if(j > 1 && i > 1) {
                    htrans += Math.abs((tsa[i - 2] - tsb[j - 2]));
                }
                dist0 = D[i - 1][j - 1] + nu * htrans + D[i][j];
                dmin = dist0;
                if(i > 1) {
                    htrans = ((tsa[i - 1] - tsa[i - 2]));
                } else {
                    htrans = tsa[i - 1];
                }
                dist = Di1[i] + D[i - 1][j] + lambda + nu * htrans;
                if(dmin > dist) {
                    dmin = dist;
                }
                if(j > 1) {
                    htrans = (tsb[j - 1] - tsb[j - 2]);
                } else {
                    htrans = tsb[j - 1];
                }
                dist = Dj1[j] + D[i][j - 1] + lambda + nu * htrans;
                if(dmin > dist) {
                    dmin = dist;
                }
                D[i][j] = dmin;
            }
            double min = Double.POSITIVE_INFINITY;
            for(int m = 0; m < D[i].length; m++) {
                min = Math.min(min, D[i][m]);
            }
            if(min > limit) {
                return Double.POSITIVE_INFINITY;
            }
        }
        dist = D[r][c];
        return dist;
    }

}
