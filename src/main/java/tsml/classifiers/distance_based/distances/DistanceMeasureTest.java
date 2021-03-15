package tsml.classifiers.distance_based.distances;

import experiments.data.DatasetLoading;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.dtw.spaces.DTWDistanceParams;
import tsml.classifiers.distance_based.distances.dtw.spaces.DTWDistanceSpace;
import tsml.classifiers.distance_based.distances.ed.EDistance;
import tsml.classifiers.distance_based.distances.ed.spaces.EDistanceParams;
import tsml.classifiers.distance_based.distances.erp.ERPDistance;
import tsml.classifiers.distance_based.distances.erp.spaces.ERPDistanceParams;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.lcss.spaces.LCSSDistanceParams;
import tsml.classifiers.distance_based.distances.msm.MSMDistance;
import tsml.classifiers.distance_based.distances.msm.spaces.MSMDistanceParams;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.twed.TWEDistance;
import tsml.classifiers.distance_based.distances.twed.spaces.TWEDistanceParams;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import tsml.classifiers.distance_based.distances.wdtw.spaces.WDTWDistanceParams;
import tsml.classifiers.distance_based.distances.wdtw.spaces.WDTWDistanceSpace;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
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

import java.io.*;
import java.util.*;

import static experiments.data.DatasetLoading.*;
import static tsml.classifiers.distance_based.distances.DistanceMeasure.DISTANCE_MEASURE_FLAG;
import static tsml.classifiers.distance_based.distances.dtw.spaces.DDTWDistanceSpace.newDDTWDistance;
import static tsml.classifiers.distance_based.distances.wdtw.spaces.WDDTWDistanceSpace.newWDDTWDistance;

@RunWith(Parameterized.class)
public class DistanceMeasureTest {
    
    @Parameterized.Parameters(name = "{3} - {2}")
    public static Collection<Object[]> data() {
        final Object[][] distanceMeasures = new Object[][]{
                {new DTWDistance(), new DTWDistanceParams()},
                {new ERPDistance(), new ERPDistanceParams()},
                {new LCSSDistance(), new LCSSDistanceParams()},
                {new MSMDistance(), new MSMDistanceParams()},
                {new EDistance(), new EDistanceParams()},
                {new WDTWDistance(), new WDTWDistanceParams()},
                {new TWEDistance(), new TWEDistanceParams()},
                {newWDDTWDistance(), new WDTWDistanceSpace()},
                {newDDTWDistance(), new DTWDistanceSpace()},
        };
        final ArrayList<Object[]> data = new ArrayList<>();
        for(final Object[] distanceMeasure : distanceMeasures) {
            for(String datasetName : Arrays.asList(
                    "ItalyPowerDemand", 
                    "GunPoint",
                    "BasicMotions"
            )) {
                final ArrayList<Object> values = new ArrayList<>(Arrays.asList(distanceMeasure));
                values.add(datasetName);
                values.add(((DistanceMeasure) values.get(0)).getName());
                data.add(values.toArray(new Object[0]));
            }
        }
        return data;
    }

    @Parameterized.Parameter(0)
    public DistanceMeasure distanceMeasure;
    
    @Parameterized.Parameter(1)
    public ParamSpaceBuilder paramSpaceBuilder;
    
    @Parameterized.Parameter(2)
    public String datasetName;
    
    @Parameterized.Parameter(3)
    public String testLabel; // dummy variable for the naming of parameterised tests
    
    private static final String TEST_RESULTS_DIR_PATH = "src/main/java/" + DistanceMeasureTest.class.getPackage().getName().replaceAll("\\.", "/") + "/test_results/";
    // switch this to true to create test results. False will run the tests instead. Creating new test results will
    // always assert fail at the end to ensure this does not get left as "true" and pass tests when it shouldn't
    private static boolean CREATE_TEST_RESULTS = false;

    public static void main(String[] args) {
        // Run main to create test results. Run Junit tests normally to test those results
        CREATE_TEST_RESULTS = true;
        JUnitCore.main(DistanceMeasureTest.class.getCanonicalName());
    }
    
    public void createTestResults(final int numTests) throws Exception {
        
        // generate the param space
        final ParamSpace paramSpace = paramSpaceBuilder.build(data);
        
        // setup a random search through parameters
        final Random random = new Random(0);
        final RandomSearch randomSearch = new RandomSearch();
        randomSearch.setRandom(random);
        randomSearch.setWithReplacement(true);
        randomSearch.buildSearch(paramSpace);
        randomSearch.setIterationLimit(numTests);

        FileUtils.makeParentDir(testResultsFilePath);
        try(final BufferedWriter writer = new BufferedWriter(new FileWriter(testResultsFilePath))) {
            writer.write("i,j,distance,params"); // writer the header line
            writer.write("\n");
            
            for(int i = 0; i < numTests; i++) {

                // indices of the two insts in the dataset
                final int instAIndex = random.nextInt(data.numInstances());
                int instBIndex; // choose a different inst to compare against
                do {
                    instBIndex = random.nextInt(data.numInstances());
                } while(instAIndex == instBIndex);
                writer.write(String.valueOf(instAIndex));
                writer.write(",");
                writer.write(String.valueOf(instBIndex));
                final TimeSeriesInstance instA = data.get(instAIndex);
                final TimeSeriesInstance instB = data.get(instBIndex);

                // last field is the parameters for the distance measure in json format
                // randomly pick a param set
                if(!randomSearch.hasNext()) throw new IllegalStateException("no remaining params");
                ParamSet paramSet = randomSearch.next();

                // clone the distance measure and apply parameters. This ensures no parameter crossover between subsequent tests
                // (shouldn't happen anyway but it's safer this way!)
                distanceMeasure = CopierUtils.deepCopy(this.distanceMeasure);
                distanceMeasure.setParams(paramSet);

                // compute the distance using the distance measure and compare to precomputed distance from results file
                final double distance = distanceMeasure.distance(instA, instB);
                writer.write(",");
                writer.write(String.valueOf(distance));
                
                // convert params to json
                writer.write(",");
                // somewhat hacky work around for distance measures which are wrapped in a transformer. For simplicity, we're only recording the parameters for the distance measure, not the wrapper itself
                if(distanceMeasure instanceof TransformDistanceMeasure) {
                    paramSet = ((TransformDistanceMeasure) distanceMeasure).getDistanceMeasure().getParams();
                }
                writer.write(paramSet.toJson());
                
                writer.write("\n");
            }

        }

        System.out.println("created tests results for " + distanceMeasure.getName() + " on " + datasetName);
    }
    
    private String dataPath;
    private String testResultsDirPath;
    private String testResultsFilePath;
    private TimeSeriesInstances data;
    private int i = 0;
    
    @Before
    public void before() {

        // load in the data
        dataPath = Arrays.asList(BAKED_IN_MTSC_DATASETS).contains(datasetName) ? BAKED_IN_MTSC_DATA_PATH :
                           BAKED_IN_TSC_DATA_PATH;
        data = loadData(dataPath, datasetName);

        // setup results file
        testResultsDirPath = TEST_RESULTS_DIR_PATH + "/" + datasetName;
        testResultsFilePath = testResultsDirPath + "/" + distanceMeasure.getName() + ".csv";

    }
    
    @Test
    public void test() throws Exception {
        i = 0;
        if(CREATE_TEST_RESULTS) {
            createTestResults(1000);
        } else {
            testDistances();
        }
    }
    
    private void testDistances() throws Exception {
        try(final BufferedReader reader = new BufferedReader(new FileReader(testResultsFilePath))) {
            reader.readLine(); // read the header line

            // for each line in the results file
            String line;
            while((line = reader.readLine()) != null) {
//                System.out.println(i);
                
                // split line on the comma
                final String[] fields = line.split(",");
                // indices of the two insts in the dataset
                final int instAIndex = Integer.parseInt(fields[0].trim());
                final int instBIndex = Integer.parseInt(fields[1].trim());
                TimeSeriesInstance instA = data.get(instAIndex);
                TimeSeriesInstance instB = data.get(instBIndex);

                // the distance
                final double targetDistance = Double.parseDouble(fields[2].trim());
                if(targetDistance == Double.POSITIVE_INFINITY) throw new IllegalArgumentException("target distance pos inf");
                if(targetDistance == Double.NEGATIVE_INFINITY) throw new IllegalArgumentException("target distance neg inf");
                if(Double.isNaN(targetDistance)) throw new IllegalArgumentException("target distance nan");

                // last field is the parameters for the distance measure in json format
                // load the json str into a paramset
                // BEWARE: super hacky fix to combine any remaining fields together into the json string. I.e. if there's a comma in the json (which with <1 parameter there defo is) then the json gets split
                ParamSet paramSet = ParamSet.fromJson(StrUtils.join(",", Arrays.copyOfRange(fields, 3, fields.length)));
                if(distanceMeasure instanceof TransformDistanceMeasure) {
                    paramSet = new ParamSet().add(DISTANCE_MEASURE_FLAG, ((TransformDistanceMeasure) distanceMeasure).getDistanceMeasure(), paramSet);
                }
                distanceMeasure.setParams(paramSet);

                // compute the distance using the distance measure and compare to precomputed distance from results file
                if(distanceMeasure instanceof MatrixBasedDistanceMeasure) {
                    ((MatrixBasedDistanceMeasure) distanceMeasure).setRecordCostMatrix(false);
                }
                final double distance = distanceMeasure.distance(instA, instB);
                Assert.assertEquals(targetDistance, distance, 0d);
                Assert.assertTrue(distance >= 0);

                // compute the distance again, this time with a limit attempting to trigger early abandon.
                // early abandon will return a distance == the full distance || distance == pos inf.
                // if the distance measure does not utilise early abandon then it should return the same distance.
                // a limit of half the unlimited distance should be sufficient.
                final double abandonedDistance = distanceMeasure.distance(instA, instB, distance / 2);
                Assert.assertTrue(abandonedDistance == Double.POSITIVE_INFINITY || abandonedDistance == distance);

                // somewhat hacky check against old implementations of each distance measure. Loosely, we check if the 
                // distance measure has an old implementation, run that and compare
                final List<Double> altDistances = altDistance(instA, instB);
                for(Double altDistance : altDistances) {
                    if(altDistance != null) {
                        if(altDistance != targetDistance) {
                            System.out.println();
                        }
                        Assert.assertEquals(targetDistance, altDistance, 0d);
                    }
                }
                
                // check that distance is same if distance measure is symmetric
                if(distanceMeasure.isSymmetric()) {
                    final double altDistance = distanceMeasure.distance(instB, instA);
                    Assert.assertEquals(distance, altDistance, 0d);
                }
                
                // check recording the cost matrix doesn't alter the distance
                if(distanceMeasure instanceof MatrixBasedDistanceMeasure) {
                    ((MatrixBasedDistanceMeasure) distanceMeasure).setRecordCostMatrix(true);
                    final double altDistance = distanceMeasure.distance(instA, instB);
                    Assert.assertEquals(distance, altDistance, 0d);
                }
                
                i++;
            }   
        }
    }
    
    private static TimeSeriesInstances loadData(String dirPath, String datasetName) {
        final String trainPath = dirPath + "/" + datasetName + "/" + datasetName + "_TRAIN.arff";
        final String testPath = dirPath + "/" + datasetName + "/" + datasetName + "_TEST.arff";
        final TimeSeriesInstances data = Converter.fromArff(DatasetLoading.loadData(trainPath));
        data.addAll(Converter.fromArff(DatasetLoading.loadData(testPath)));
        return data;
    }
    
    // find an alternative distance value from a different implementation
    private List<Double> altDistance(TimeSeriesInstance inst1, TimeSeriesInstance inst2) {
        Double oldDistance = null;
        Double origDistance = null;
        Double monashDistance = null;
        final int len = inst1.getMaxLength();
        final String name = distanceMeasure.getName().replace("Distance", "");
        final double window;
        switch(name) {
            case "ERP":
                window = ((ERPDistance) distanceMeasure).getWindow();
                break;
            case "DTW":
                window = ((DTWDistance) distanceMeasure).getWindow();
                break;
            case "DDTW":
                window = ((DTWDistance) ((TransformDistanceMeasure) distanceMeasure).getDistanceMeasure())
                                 .getWindow();
                break;
            case "LCSS":
                window = ((LCSSDistance) distanceMeasure).getWindow();
                break;
            default:
                window = Double.NaN;
                break;
        }
        final int floorWindow = (int) Math.floor(window * len);
        final double floorWindowProportional = (double) floorWindow / len;
        
        if(!inst1.isMultivariate() && !inst2.isMultivariate()) {


            // add the previous implementations
            if("LCSS".equals(name)) {
                oldDistance = PreviousDistanceMeasureVersions.lcss(Converter.toArff(inst1), Converter.toArff(inst2),
                        Double.POSITIVE_INFINITY,
                        floorWindow,
                        ((LCSSDistance) distanceMeasure).getEpsilon());
            } else if("DDTW".equals(name)) {
                oldDistance = PreviousDistanceMeasureVersions.dtw(Converter.toArff(new Derivative().transform(inst1)),
                        Converter.toArff(new Derivative().transform(inst2)),
                        Double.POSITIVE_INFINITY,
                        floorWindow);
            } else if("WDDTW".equals(name)) {
                oldDistance = PreviousDistanceMeasureVersions.wdtw(Converter.toArff(new Derivative().transform(inst1)),
                        Converter.toArff(new Derivative().transform(inst2)),
                        Double.POSITIVE_INFINITY,
                        ((WDTWDistance) ((TransformDistanceMeasure) distanceMeasure)
                                                .getDistanceMeasure()).getG());
            } else if("DTW".equals(name)) {
                oldDistance = PreviousDistanceMeasureVersions.dtw(Converter.toArff(inst1), Converter.toArff(inst2),
                        Double.POSITIVE_INFINITY,
                        floorWindow);
            } else if("WDTW".equals(name)) {
                oldDistance = PreviousDistanceMeasureVersions.wdtw(Converter.toArff(inst1), Converter.toArff(inst2),
                        Double.POSITIVE_INFINITY,
                        ((WDTWDistance) distanceMeasure).getG());
            } else if("ERP".equals(name)) {
                oldDistance = PreviousDistanceMeasureVersions.erp(Converter.toArff(inst1), Converter.toArff(inst2),
                        Double.POSITIVE_INFINITY,
                        floorWindow, ((ERPDistance) distanceMeasure).getG());
            } else if("MSM".equals(name)) {
                oldDistance = PreviousDistanceMeasureVersions.msm(Converter.toArff(inst1), Converter.toArff(inst2),
                        Double.POSITIVE_INFINITY,
                        ((MSMDistance) distanceMeasure).getC());
            } else if("TWED".equals(name)) {
                oldDistance = PreviousDistanceMeasureVersions.twed(Converter.toArff(inst1), Converter.toArff(inst2),
                        Double.POSITIVE_INFINITY,
                        ((TWEDistance) distanceMeasure).getLambda(),
                        ((TWEDistance) distanceMeasure).getNu());
            }
            
            // convert to raw arrays
            final double[] a = inst1.toValueArray()[0];
            final double[] b = inst2.toValueArray()[0];
            
            // find the orig distances
            if("LCSS".equals(name)) {
                origDistance = DistanceMeasuresFromBitbucket
                                       .lcss(a, b, ((LCSSDistance) distanceMeasure).getEpsilon(), floorWindow);
            } else if("DTW".equals(name)) {
                origDistance = DistanceMeasuresFromBitbucket.dtw(a, b, Double.POSITIVE_INFINITY, window);
            } else if("WDTW".equals(name)) {
                origDistance = DistanceMeasuresFromBitbucket.wdtw(a, b, Double.POSITIVE_INFINITY, ((WDTWDistance) distanceMeasure).getG());
            } else if("ERP".equals(name)) {
                origDistance = DistanceMeasuresFromBitbucket.erp(a, b, ((ERPDistance) distanceMeasure).getG(), floorWindowProportional);
            } else if("MSM".equals(name)) {
                origDistance = DistanceMeasuresFromBitbucket.msm(a, b, ((MSMDistance) distanceMeasure).getC());
            } else if("TWED".equals(name)) {
                origDistance = DistanceMeasuresFromBitbucket.twed(a, b,
                        ((TWEDistance) distanceMeasure).getNu(), ((TWEDistance) distanceMeasure).getLambda());
            }
            
            // find the monash implementation distance
            if("LCSS".equals(name)) {
                monashDistance = MonashDistanceMeasures.lcss(a, b, Double.POSITIVE_INFINITY, floorWindow, ((LCSSDistance) distanceMeasure).getEpsilon());
            } else if("DDTW".equals(name)) {
                monashDistance = MonashDistanceMeasures.ddtw(a, b, Double.POSITIVE_INFINITY, floorWindow);
            } else if("WDDTW".equals(name)) {
                monashDistance = MonashDistanceMeasures.wddtw(a, b, ((WDTWDistance) ((TransformDistanceMeasure) distanceMeasure)
                                                                                            .getDistanceMeasure()).getG());
            } else if("DTW".equals(name)) {
                monashDistance = MonashDistanceMeasures.dtw(a, b, Double.POSITIVE_INFINITY, floorWindow);
            } else if("WDTW".equals(name)) {
                monashDistance = MonashDistanceMeasures.wdtw(a, b, ((WDTWDistance) distanceMeasure).getG());
            } else if("ERP".equals(name)) {
                monashDistance = MonashDistanceMeasures.erp(a, b, floorWindow, ((ERPDistance) distanceMeasure).getG());
            } else if("MSM".equals(name)) {
                monashDistance = MonashDistanceMeasures.msm(a, b, Double.POSITIVE_INFINITY, ((MSMDistance) distanceMeasure).getC());
            } else if("TWED".equals(name)) {
                monashDistance = MonashDistanceMeasures.twed(a, b, ((TWEDistance) distanceMeasure).getNu(), ((TWEDistance) distanceMeasure).getLambda());
            } else if("E".equals(name)) {
                monashDistance = MonashDistanceMeasures.ed(a, b, Double.POSITIVE_INFINITY);
            }
        }
        
        
        return Arrays.asList(oldDistance, origDistance, monashDistance);
    }

    private static class DistanceMeasuresFromBitbucket {

        // distance measure code snapshot from bitbucket to github move
        // https://github.com/uea-machine-learning/tsml/tree/29b5558ebab6b5dd427ed45d028f52f6e9401e30
        
        public static double dtw(double[] a, double[] b, double cutoff, double r) {
            
            double minDist;
            boolean tooBig;
            // Set the longest series to a. is this necessary?
            double[] temp;
            if(a.length<b.length){
                temp=a;
                a=b;
                b=temp;
            }
            int n=a.length;
            int m=b.length;
/*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp 
generalised for variable window size
* */
            int windowSize=(int)(r*n);   //Rounded down.
            //No Warp, windowSize=1
            if(windowSize<1) windowSize=1;
                //Full Warp : windowSize=n, otherwise scale between		
            else if(windowSize<n)
                windowSize++;
            
            double[][] matrixD = null;
            //Extra memory than required, could limit to windowsize,
            //        but avoids having to recreate during CV 
            //for varying window sizes        
            if(matrixD==null)
                matrixD=new double[n][m];
        
/*
//Set boundary elements to max. 
*/
            int start,end;
            for(int i=0;i<n;i++){
                start=windowSize<i?i-windowSize:0;
                end=i+windowSize+1<m?i+windowSize+1:m;
                for(int j=start;j<end;j++)
                    matrixD[i][j]=Double.MAX_VALUE;
            }
            matrixD[0][0]=(a[0]-b[0])*(a[0]-b[0]);
            //a is the longer series. 
            //Base cases for warping 0 to all with max interval	r	
            //Warp a[0] onto all b[1]...b[r+1]
            for(int j=1;j<windowSize && j<m;j++)
                matrixD[0][j]=matrixD[0][j-1]+(a[0]-b[j])*(a[0]-b[j]);

            //	Warp b[0] onto all a[1]...a[r+1]
            for(int i=1;i<windowSize && i<n;i++)
                matrixD[i][0]=matrixD[i-1][0]+(a[i]-b[0])*(a[i]-b[0]);
            //Warp the rest,
            for (int i=1;i<n;i++){
                tooBig=true;
                start=windowSize<i?i-windowSize+1:1;
                end=i+windowSize<m?i+windowSize:m;
                for (int j = start;j<end;j++){
                    minDist=matrixD[i][j-1];
                    if(matrixD[i-1][j]<minDist)
                        minDist=matrixD[i-1][j];
                    if(matrixD[i-1][j-1]<minDist)
                        minDist=matrixD[i-1][j-1];
                    matrixD[i][j]=minDist+(a[i]-b[j])*(a[i]-b[j]);
                    if(tooBig&&matrixD[i][j]<cutoff)
                        tooBig=false;
                }
                //Early abandon
                if(tooBig){
                    return Double.MAX_VALUE;
                }
            }
            //Find the minimum distance at the end points, within the warping window. 
            return matrixD[n-1][m-1];
        }

        private static class NumberVector{

            private double[] values;
            public NumberVector(double[] values){
                this.values = values;
            }

            public int getDimensionality(){
                return values.length;
            }

            public double doubleValue(int dimension){
                try{
                    return values[dimension - 1];
                }catch(IndexOutOfBoundsException e) {
                    throw new IllegalArgumentException("Dimension " + dimension + " out of range.");
                }
            }
        }
        
        public static double lcss(double[] first, double[] second, double epsilon, int delta) {

            double[] a  = first;
            double[] b = second;
            int m = first.length;
            int n = second.length;

            int[][] lcss = new int[m+1][n+1];
            int[][] lastX = new int[m+1][n+1];
            int[][] lastY = new int[m+1][n+1];


            for(int i = 0; i < m; i++){
                for(int j = i-delta; j <= i+delta; j++){
                    //                System.out.println("here");
                    if(j < 0 || j >= n){
                        //do nothing
                    }else if(b[j]+epsilon >= a[i] && b[j]-epsilon <=a[i]){
                        lcss[i+1][j+1] = lcss[i][j]+1;
                        lastX[i+1][j+1] = i;
                        lastY[i+1][j+1] = j;
                    }else if(lcss[i][j+1] > lcss[i+1][j]){
                        lcss[i+1][j+1] = lcss[i][j+1];
                        lastX[i+1][j+1] = i;
                        lastY[i+1][j+1] = j+1;
                    }else if(lcss[i][j+1] < lcss[i+1][j]){
                        lcss[i+1][j+1] = lcss[i+1][j];
                        lastX[i+1][j+1] = i+1;
                        lastY[i+1][j+1] = j;
                    } else {
                        // take the max of left or top. topLeft has no effect as always equal to or less than left or top
                        lcss[i+1][j+1] = Math.max(lcss[i][j], Math.max(lcss[i][j + 1], lcss[i + 1][j]));
                        lastX[i+1][j+1] = i;
                        lastY[i+1][j+1] = j;
                    }
                    
                    // orig bugged version
//                    if(j < 0 || j >= n){
//                        //do nothing
//                    }else if(b[j]+epsilon >= a[i] && b[j]-epsilon <=a[i]){
//                        lcss[i+1][j+1] = lcss[i][j]+1;
//                        lastX[i+1][j+1] = i;
//                        lastY[i+1][j+1] = j;
//                    }else if(lcss[i][j+1] > lcss[i+1][j]){
//                        lcss[i+1][j+1] = lcss[i][j+1];
//                        lastX[i+1][j+1] = i;
//                        lastY[i+1][j+1] = j+1;
//                    }else{
//                        lcss[i+1][j+1] = lcss[i+1][j];
//                        lastX[i+1][j+1] = i+1;
//                        lastY[i+1][j+1] = j;
//                    }
                }
            }

            int max = -1;
            for(int i = 1; i < lcss[lcss.length-1].length; i++){
                if(lcss[lcss.length-1][i] > max){
                    max = lcss[lcss.length-1][i];
                }
            }
            return 1-((double)max/m);
        }
        
        public static double erp(double[] first, double[] second, double g, double bandSize) {
            // Current and previous columns of the matrix
            
            NumberVector v1 = new NumberVector(first);
            NumberVector v2 = new NumberVector(second);
            
            double[] curr = new double[v2.getDimensionality()];
            double[] prev = new double[v2.getDimensionality()];

            // size of edit distance band
            // bandsize is the maximum allowed distance to the diagonal
            //        int band = (int) Math.ceil(v2.getDimensionality() * bandSize);
            int band = (int) Math.ceil(v2.getDimensionality() * bandSize);

            // g parameter for local usage
            double gValue = g;

            for (int i = 0; i < v1.getDimensionality(); i++) {
                // Swap current and prev arrays. We'll just overwrite the new curr.
                {
                    double[] temp = prev;
                    prev = curr;
                    curr = temp;
                }
                int l = i - (band + 1);
                if (l < 0) {
                    l = 0;
                }
                int r = i + (band + 1);
                if (r > (v2.getDimensionality() - 1)) {
                    r = (v2.getDimensionality() - 1);
                }

                for (int j = l; j <= r; j++) {
                    if (Math.abs(i - j) <= band) {
                        // compute squared distance of feature vectors
                        double val1 = v1.doubleValue(i + 1);
                        double val2 = gValue;
                        double diff = (val1 - val2);
                        final double d1 = Math.sqrt(diff * diff);

                        val1 = gValue;
                        val2 = v2.doubleValue(j + 1);
                        diff = (val1 - val2);
                        final double d2 = Math.sqrt(diff * diff);

                        val1 = v1.doubleValue(i + 1);
                        val2 = v2.doubleValue(j + 1);
                        diff = (val1 - val2);
                        final double d12 = Math.sqrt(diff * diff);

                        final double dist1 = d1 * d1;
                        final double dist2 = d2 * d2;
                        final double dist12 = d12 * d12;

                        final double cost;

                        if ((i + j) != 0) {
                            if ((i == 0) || ((j != 0) && (((prev[j - 1] + dist12) > (curr[j - 1] + dist2)) && ((curr[j - 1] + dist2) < (prev[j] + dist1))))) {
                                // del
                                cost = curr[j - 1] + dist2;
                            } else if ((j == 0) || ((i != 0) && (((prev[j - 1] + dist12) > (prev[j] + dist1)) && ((prev[j] + dist1) < (curr[j - 1] + dist2))))) {
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
                        // steps[i][j] = step;
                    } else {
                        curr[j] = Double.POSITIVE_INFINITY; // outside band
                    }
                }
            }

            return curr[v2.getDimensionality() - 1];
        }
        
        public static double twed(double[] a, double[] b, double nu, double lambda) {
            int dim=1;
            double dist, disti1, distj1;
            double[][] ta=new double[a.length][dim];
            double[][] tb=new double[a.length][dim];
            double[] tsa=new double[a.length];
            double[] tsb=new double[b.length];
            for(int i=0;i<tsa.length;i++)
                tsa[i]=(i+1);
            for(int i=0;i<tsb.length;i++)
                tsb[i]=(i+1);

            int r = ta.length;
            int c = tb.length;
            int i,j,k;
            //Copy over values
            for(i=0;i<a.length;i++)
                ta[i][0]=a[i];
            for(i=0;i<b.length;i++)
                tb[i][0]=b[i];

        /* allocations in c
	double **D = (double **)calloc(r+1, sizeof(double*));
	double *Di1 = (double *)calloc(r+1, sizeof(double));
	double *Dj1 = (double *)calloc(c+1, sizeof(double));
	for(i=0; i<=r; i++) {
		D[i]=(double *)calloc(c+1, sizeof(double));
	}
*/
            double [][]D = new double[r+1][c+1];
            double[] Di1 = new double[r+1];
            double[] Dj1 = new double[c+1];
            // local costs initializations
            for(j=1; j<=c; j++) {
                distj1=0;
                for(k=0; k<dim; k++)
                    if(j>1){
                        //CHANGE AJB 8/1/16: Only use power of 2 for speed up,                       
                        distj1+=(tb[j-2][k]-tb[j-1][k])*(tb[j-2][k]-tb[j-1][k]);
                        // OLD VERSION                    distj1+=Math.pow(Math.abs(tb[j-2][k]-tb[j-1][k]),degree);
                        // in c:               distj1+=pow(fabs(tb[j-2][k]-tb[j-1][k]),degree);
                    }
                    else
                        distj1+=tb[j-1][k]*tb[j-1][k];
                //OLD              		distj1+=Math.pow(Math.abs(tb[j-1][k]),degree);
                Dj1[j]=(distj1);
            }

            for(i=1; i<=r; i++) {
                disti1=0;
                for(k=0; k<dim; k++)
                    if(i>1)
                        disti1+=(ta[i-2][k]-ta[i-1][k])*(ta[i-2][k]-ta[i-1][k]);
                        // OLD                 disti1+=Math.pow(Math.abs(ta[i-2][k]-ta[i-1][k]),degree);
                    else
                        disti1+=(ta[i-1][k])*(ta[i-1][k]);
                //OLD                  disti1+=Math.pow(Math.abs(ta[i-1][k]),degree);

                Di1[i]=(disti1);

                for(j=1; j<=c; j++) {
                    dist=0;
                    for(k=0; k<dim; k++){
                        dist+=(ta[i-1][k]-tb[j-1][k])*(ta[i-1][k]-tb[j-1][k]);
                        //                  dist+=Math.pow(Math.abs(ta[i-1][k]-tb[j-1][k]),degree);
                        if(i>1&&j>1)
                            dist+=(ta[i-2][k]-tb[j-2][k])*(ta[i-2][k]-tb[j-2][k]);
                        //                    dist+=Math.pow(Math.abs(ta[i-2][k]-tb[j-2][k]),degree);
                    }
                    D[i][j]=(dist);
                }
            }// for i

            // border of the cost matrix initialization
            D[0][0]=0;
            for(i=1; i<=r; i++)
                D[i][0]=D[i-1][0]+Di1[i];
            for(j=1; j<=c; j++)
                D[0][j]=D[0][j-1]+Dj1[j];

            double dmin, htrans, dist0;
            int iback;

            for (i=1; i<=r; i++){
                for (j=1; j<=c; j++){
                    htrans=Math.abs((tsa[i-1]-tsb[j-1]));
                    if(j>1&&i>1)
                        htrans+=Math.abs((tsa[i-2]-tsb[j-2]));
                    dist0=D[i-1][j-1]+nu*htrans+D[i][j];
                    dmin=dist0;
                    if(i>1)
                        htrans=((tsa[i-1]-tsa[i-2]));
                    else htrans=tsa[i-1];
                    dist=Di1[i]+D[i-1][j]+lambda+nu*htrans;
                    if(dmin>dist){
                        dmin=dist;
                    }
                    if(j>1)
                        htrans=(tsb[j-1]-tsb[j-2]);
                    else htrans=tsb[j-1];
                    dist=Dj1[j]+D[i][j-1]+lambda+nu*htrans;
                    if(dmin>dist){
                        dmin=dist;
                    }
                    D[i][j] = dmin;
                }
            }

            dist = D[r][c];
            return dist;
        }
        
        private static double[] initWeights(int seriesLength, double g){
            double[] weightVector = new double[seriesLength];
            double halfLength = (double)seriesLength/2;

            for(int i = 0; i < seriesLength; i++){
                weightVector[i] = 1/(1+Math.exp(-g*(i-halfLength)));
            }
            return weightVector;
        }
        
        public static double wdtw(double[] first, double[] second, double cutOffValue, double g) {

            double[] weightVector = null;
            
            if(weightVector==null){
                weightVector = initWeights(first.length, g);
            }
            double[][] distances;

            //create empty array
            distances = new double[first.length][second.length];

            //first value
            distances[0][0] = weightVector[0]*((first[0]-second[0])*(first[0]-second[0]));

            //early abandon if first values is larger than cut off
            if(distances[0][0] > cutOffValue){
                return Double.MAX_VALUE;
            }

            //top row
            for(int i=1;i<second.length;i++){
                distances[0][i] = distances[0][i-1]+weightVector[i]*((first[0]-second[i])*(first[0]-second[i])); //edited by Jay
            }

            //first column
            for(int i=1;i<first.length;i++){
                distances[i][0] = distances[i-1][0]+weightVector[i]*((first[i]-second[0])*(first[i]-second[0])); //edited by Jay
            }

            //warp rest
            double minDistance;
            for(int i = 1; i<first.length; i++){
                boolean overflow = true;

                for(int j = 1; j<second.length; j++){
                    //calculate distances
                    minDistance = Math.min(distances[i][j-1], Math.min(distances[i-1][j], distances[i-1][j-1]));
                    distances[i][j] = minDistance+weightVector[Math.abs(i-j)] *((first[i]-second[j])*(first[i]-second[j])); //edited by Jay

                    if(overflow && distances[i][j] < cutOffValue){
                        overflow = false; // because there's evidence that the path can continue
                    }
                    //                    
                    //                if(minDistance > cutOffValue && isEarlyAbandon){
                    //                    distances[i][j] = Double.MAX_VALUE;
                    //                }else{
                    //                    distances[i][j] = minDistance+weightVector[Math.abs(i-j)] *(first[i]-second[j])*(first[i]-second[j]); //edited by Jay
                    //                    overflow = false;
                    //                }
                }

                //early abandon
                if(overflow){
                    return Double.MAX_VALUE;
                }
            }
            return distances[first.length-1][second.length-1];
        }
        
        public static double msm(double[] a, double[] b, double c) {

            int m, n, i, j;
            m = a.length;
            n = b.length;

            double[][] cost = new double[m][n];

            // Initialization
            cost[0][0] = Math.abs(a[0] - b[0]);

            for (i = 1; i< m; i++) {
                cost[i][0] = cost[i-1][0] + editCost(a[i], a[i-1], b[0], c);
            }

            for (j = 1; j < n; j++) {
                cost[0][j] = cost[0][j-1] + editCost(b[j], a[0], b[j-1], c);
            }

            // Main Loop
            for( i = 1; i < m; i++){
                for ( j = 1; j < n; j++){
                    double d1,d2, d3;
                    d1 = cost[i-1][j-1] + Math.abs(a[i] - b[j] );
                    d2 = cost[i-1][j] + editCost(a[i], a[i-1], b[j], c);
                    d3 = cost[i][j-1] + editCost(b[j], a[i], b[j-1], c);
                    cost[i][j] = Math.min( d1, Math.min(d2,d3) );
                }
            }

            // Output
            return cost[m-1][n-1];
        }


        private static double editCost( double new_point, double x, double y, double c){
            double dist = 0;

            if ( ( (x <= new_point) && (new_point <= y) ) ||
                         ( (y <= new_point) && (new_point <= x) ) ) {
                dist = c;
            }
            else{
                dist = c + Math.min( Math.abs(new_point - x) , Math.abs(new_point - y) );
            }

            return dist;
        }
    }
    
    private static class PreviousDistanceMeasureVersions {

        // this is the code for the distance measures BEFORE the big overhaul to variable length and row-by-row variants for increased speed
        
        public static double erp(Instance first, Instance second, double limit, int band, double penalty) {
    
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

        public static double wdtw(Instance a, Instance b, double limit, double g) {
    
            int aLength = a.numAttributes() - 1;
            int bLength = b.numAttributes() - 1;
    
            double[] weightVector = generateWeights(aLength, g);
    
            //create empty array
            double[][] distances = new double[aLength][bLength];
    
            //first value
            distances[0][0] = (a.value(0) - b.value(0)) * (a.value(0) - b.value(0)) * weightVector[0];
    
    
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

        public static double dtw(Instance first, Instance second, double limit, int windowSize) {
    
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

        public static double msm(Instance a, Instance b, double limit, double c) {
    
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

        public static double twed(Instance a, Instance b, double limit, double lambda, double nu) {
    
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

        public static double lcss(Instance a, Instance b, double limit, int delta, double epsilon) {
    
            int aLength = a.numAttributes() - 1;
            int bLength = b.numAttributes() - 1;
    
            // 22/10/19 goastler - limit LCSS such that if any value in the current window is larger than the limit then we can stop here, no point in doing the extra work
            if(limit != Double.POSITIVE_INFINITY) { // check if there's a limit set
                // if so then reverse engineer the max LCSS distance and replace the limit
                // this is just the inverse of the return value integer rounded to an LCSS distance
                limit = (int) ((1 - limit) * aLength) + 1;
            }
    
            int[][] lcss = new int[aLength + 1][bLength + 1];
    
            int warpingWindow = delta;
            if(warpingWindow < 0) {
                warpingWindow = aLength + 1;
            }
    
            for(int i = 0; i < aLength; i++) {
                boolean tooBig = true;
                for(int j = i - warpingWindow; j <= i + warpingWindow; j++) {
                    if(j < 0) {
                        j = -1;
                    } else if(j >= bLength) {
                        j = i + warpingWindow;
                    } else {
                        if(b.value(j) + epsilon >= a.value(i) && b.value(j) - epsilon <= a
                                                                                                 .value(i)) {
                            lcss[i + 1][j + 1] = lcss[i][j] + 1;
                            //                    } else if(lcss[i][j + 1] > lcss[i + 1][j]) {
                            //                        lcss[i + 1][j + 1] = lcss[i][j + 1];
                            //                    } else {
                            //                        lcss[i + 1][j + 1] = lcss[i + 1][j];
                        }
                        else {
                            lcss[i + 1][j + 1] = Math.max(lcss[i + 1][j], Math.max(lcss[i][j], lcss[i][j + 1]));
                        }
                        // if this value is less than the limit then fast-fail the limit overflow
                        if(tooBig && lcss[i + 1][j + 1] <= limit) {
                            tooBig = false;
                        }
                    }
                }
    
                // if no element is lower than the limit then early abandon
                if(tooBig) {
                    return Double.POSITIVE_INFINITY;
                }
    
            }
    
            int max = -1;
            for(int j = 1; j < lcss[lcss.length - 1].length; j++) {
                if(lcss[lcss.length - 1][j] > max) {
                    max = lcss[lcss.length - 1][j];
                }
            }
            return 1 - ((double) max / aLength);
        }
    }

    private static class MonashDistanceMeasures {


        public static double min(double A, double B, double C) {
            if (A < B) {
                if (A < C) {
                    // A < B and A < C
                    return A;
                } else {
                    // C < A < B
                    return C;
                }
            } else {
                if (B < C) {
                    // B < A and B < C
                    return B;
                } else {
                    // C < B < A
                    return C;
                }
            }
        }

        public static double squaredDistance(double A, double B) {
            double x = A - B;
            return x * x;
        }
        
        public static double dtw(double[] series1, double[] series2, double bsf, int windowSize) {
            // monash's dtw doesn't work properly when the window is zero (i.e. euclidean distance).
            // default to their ED to make things work, but this is a bug in their code and therefore their PF / tsml's PF-WRAPPER

            if(windowSize == 0) {
                return ed(series1, series2, Double.POSITIVE_INFINITY);
            }
            
            if (windowSize == -1) {
                windowSize = series1.length;
            }

            int length1 = series1.length;
            int length2 = series2.length;

            int maxLength = Math.max(length1, length2);

            double[] prevRow = new double[maxLength];
            double[] currentRow = new double[maxLength];

            if (prevRow == null || prevRow.length < maxLength) {
                prevRow = new double[maxLength];
            }

            if (currentRow == null || currentRow.length < maxLength) {
                currentRow = new double[maxLength];
            }

            int i, j;
            double prevVal;
            double thisSeries1Val = series1[0];

            // initialising the first row - do this in prevRow so as to save swapping rows before next row
            prevVal = prevRow[0] = squaredDistance(thisSeries1Val, series2[0]);

            for (j = 1; j < Math.min(length2, 1 + windowSize); j++) {
                prevVal = prevRow[j] = prevVal + squaredDistance(thisSeries1Val, series2[j]);
            }

            // the second row is a special case
            if (length1 >= 2){
                thisSeries1Val = series1[1];

                if (windowSize>0){
                    currentRow[0] = prevRow[0]+squaredDistance(thisSeries1Val, series2[0]);
                }

                // in this special case, neither matrix[1][0] nor matrix[0][1] can be on the (shortest) minimum path
                prevVal = currentRow[1]=prevRow[0]+squaredDistance(thisSeries1Val, series2[1]);
                int jStop = (windowSize + 2 > length2) ? length2 : windowSize + 2;

                for (j = 2; j < jStop; j++) {
                    // for the second row, matrix[0][j - 1] cannot be on a (shortest) minimum path
                    prevVal = currentRow[j] = Math.min(prevVal, prevRow[j - 1]) + squaredDistance(thisSeries1Val, series2[j]);
                }
            }

            // third and subsequent rows
            for (i = 2; i < length1; i++) {
                int jStart;
                int jStop = (i + windowSize >= length2) ? length2-1 : i + windowSize;

                // the old currentRow becomes this prevRow and so the currentRow needs to use the old prevRow
                double[] tmp = prevRow;
                prevRow = currentRow;
                currentRow = tmp;

                thisSeries1Val = series1[i];

                if (i - windowSize < 1) {
                    jStart = 1;
                    currentRow[0] = prevRow[0] + squaredDistance(thisSeries1Val, series2[0]);
                }
                else {
                    jStart = i - windowSize;
                }

                if (jStart <= jStop){
                    // If jStart is the start of the window, [i][jStart-1] is outside the window.
                    // Otherwise jStart-1 must be 0 and the path through [i][0] can never be less than the path directly from [i-1][0]
                    prevVal = currentRow[jStart] = Math.min(prevRow[jStart - 1], prevRow[jStart])+ squaredDistance(thisSeries1Val, series2[jStart]);
                    for (j = jStart+1; j < jStop; j++) {
                        prevVal = currentRow[j] = min(prevRow[j - 1], prevVal, prevRow[j])
                                                          + squaredDistance(thisSeries1Val, series2[j]);
                    }

                    if (i + windowSize >= length2) {
                        // the window overruns the end of the sequence so can have a path through prevRow[jStop]
                        currentRow[jStop] = min(prevRow[jStop - 1], prevRow[jStop], prevVal) + squaredDistance(thisSeries1Val, series2[jStop]);
                    }
                    else {
                        currentRow[jStop] = Math.min(prevRow[jStop - 1], prevVal) + squaredDistance(thisSeries1Val, series2[jStop]);
                    }
                }
            }

            double res = currentRow[length2 - 1];

            return res;
        }

        public static double ddtw(double[] a, double[] b, double bsf, int w) {
            return dtw(getDeriv(a), getDeriv(b), bsf, w);
        }

        private static double[] getDeriv(double[] series) {
            double[] d = new double[series.length];
            for (int i = 1; i < series.length - 1 ; i++) {
                d[i] = ((series[i] - series[i - 1]) + ((series[i + 1] - series[i - 1]) / 2.0)) / 2.0;
            }

            d[0] = d[1];
            d[d.length - 1] = d[d.length - 2];

            return d;
        }
        
        public static double erp(double[] first, double[] second, int windowSize, double gValue) {

            double[] curr = null, prev = null;
            
            int m = first.length;
            int n = second.length;

            if (curr == null || curr.length < m) {
                curr = new double[m];
                prev = new double[m];
            } else {
                // FPH: init to 0 just in case, didn't check if
                // important
                for (int i = 0; i < curr.length; i++) {
                    curr[i] = 0.0;
                    prev[i] = 0.0;
                }
            }

            // size of edit distance band
            // bandsize is the maximum allowed distance to the diagonal
            // int band = (int) Math.ceil(v2.getDimensionality() *
            // bandSize);
            //		int band = (int) Math.ceil(m * bandSize);
            int band = windowSize;

            // g parameter for local usage
            for (int i = 0; i < m; i++) {
                // Swap current and prev arrays. We'll just overwrite
                // the new curr.
                {
                    double[] temp = prev;
                    prev = curr;
                    curr = temp;
                }
                int l = i - (band + 1);
                if (l < 0) {
                    l = 0;
                }
                int r = i + (band + 1);
                if (r > (m - 1)) {
                    r = (m - 1);
                }

                for (int j = l; j <= r; j++) {
                    if (Math.abs(i - j) <= band) {
                        // compute squared distance of feature
                        // vectors
                        double val1 = first[i];
                        double val2 = gValue;
                        double diff = (val1 - val2);
                        //					final double d1 = Math.sqrt(diff * diff);
                        final double d1 = diff;//FPH simplificaiton

                        val1 = gValue;
                        val2 = second[j];
                        diff = (val1 - val2);
                        //					final double d2 = Math.sqrt(diff * diff);
                        final double d2 = diff;

                        val1 = first[i];
                        val2 = second[j];
                        diff = (val1 - val2);
                        //					final double d12 = Math.sqrt(diff * diff);
                        final double d12 = diff;

                        final double dist1 = d1 * d1;
                        final double dist2 = d2 * d2;
                        final double dist12 = d12 * d12;

                        final double cost;

                        if ((i + j) != 0) {
                            if ((i == 0) || ((j != 0) && (((prev[j - 1] + dist12) > (curr[j - 1] + dist2))
                                                                  && ((curr[j - 1] + dist2) < (prev[j] + dist1))))) {
                                // del
                                cost = curr[j - 1] + dist2;
                            } else if ((j == 0) || ((i != 0) && (((prev[j - 1] + dist12) > (prev[j] + dist1))
                                                                         && ((prev[j] + dist1) < (curr[j - 1] + dist2))))) {
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
                        // steps[i][j] = step;
                    } else {
                        curr[j] = Double.POSITIVE_INFINITY; // outside
                        // band
                    }
                }
            }

            return curr[m - 1];
        }
        
        public static double ed(double[] s, double[] t, double bsf) {
            int i = 0;
            double total = 0;

            //assume s.length == t.length for this implementation
            //TODO note <=, if bsf = 0, < will cause problems when early abandoning
            for (i = 0; i < s.length & total <= bsf; i++){
                total += (s[i] - t[i]) * (s[i] - t[i]);
            }

            //		System.out.println("Euclidean: early abandon after: " + i + " from: " + s.length);

            //		return Math.sqrt(total);
            return total;
        }

        private static int sim(double a, double b, double epsilon) {
            return (Math.abs(a - b) <= epsilon) ? 1 : 0;
        }

        public static double lcss(double[] series1, double[] series2, double bsf, int windowSize, double epsilon) {
            if (windowSize == -1) {
                windowSize = series1.length;
            }

            int length1 = series1.length;
            int length2 = series2.length;

            int maxLength = Math.max(length1, length2);
            int minLength = Math.min(length1, length2);

            int [][]matrix = new int[length1][length2];
            //		int[][] matrix = MemoryManager.getInstance().getIntMatrix(0);

            int i, j;

            matrix[0][0] = sim(series1[0], series2[0], epsilon);
            for (i = 1; i < Math.min(length1, 1 + windowSize); i++) {
                matrix[i][0] = (sim(series1[i], series2[0], epsilon)==1)?sim(series1[i], series2[0], epsilon):matrix[i-1][0];
            }

            for (j = 1; j < Math.min(length2, 1 + windowSize); j++) {
                matrix[0][j] = (sim(series1[0], series2[j], epsilon)==1?sim(series1[0], series2[j], epsilon):matrix[0][j-1]);
            }

            if (j < length2)
                matrix[0][j] = Integer.MIN_VALUE;


            for (i = 1; i < length1; i++) {
                int jStart = (i - windowSize < 1) ? 1 : i - windowSize;
                int jStop = (i + windowSize + 1 > length2) ? length2 : i + windowSize + 1;

                if (i-windowSize-1>=0)
                    matrix[i][i-windowSize-1] = Integer.MIN_VALUE;
                for (j = jStart; j < jStop; j++) {
                    if (sim(series1[i], series2[j], epsilon) == 1) {
                        matrix[i][j] = matrix[i - 1][j - 1] + 1;
                    } else {
                        matrix[i][j] = max(matrix[i - 1][j - 1], matrix[i][j - 1], matrix[i - 1][j]);
                    }
                }
                if (jStop < length2)
                    matrix[i][jStop] = Integer.MIN_VALUE;
            }

            double res = 1.0 - 1.0 * matrix[length1 - 1][length2 - 1] / minLength;
            return res;
        }

        public static final int max(int A, int B, int C) {
            if (A > B) {
                if (A > C) {
                    return A;
                } else {
                    // C > A > B
                    return C;
                }
            } else {
                if (B > C) {
                    // B > A and B > C
                    return B;
                } else {
                    // C > B > A
                    return C;
                }
            }
        }
        
        public static double msm(double[] first, double[] second, double bsf, double c) {

            int m = first.length, n = second.length;
            int maxLength=(m>=n)?m:n;
            double[][]cost = new double[m][n];
            //		double[][]cost = MemoryManager.getInstance().getDoubleMatrix(0);
            if (cost == null || cost.length < m || cost[0].length < n) {
                cost = new double[m][n];
            }

            // Initialization
            cost[0][0] = Math.abs(first[0] - second[0]);
            for (int i = 1; i < m; i++) {
                cost[i][0] = cost[i - 1][0] + calcualteCost(first[i], first[i - 1], second[0], c);
            }
            for (int i = 1; i < n; i++) {
                cost[0][i] = cost[0][i - 1] + calcualteCost(second[i], first[0], second[i - 1], c);
            }

            // Main Loop
            for (int i = 1; i < m; i++) {
                for (int j = 1; j < n; j++) {
                    double d1, d2, d3;
                    d1 = cost[i - 1][j - 1] + Math.abs(first[i] - second[j]);
                    d2 = cost[i - 1][j] + calcualteCost(first[i], first[i - 1], second[j], c);
                    d3 = cost[i][j - 1] + calcualteCost(second[j], first[i], second[j - 1], c);
                    cost[i][j] = Math.min(d1, Math.min(d2, d3));

                }
            }
            // Output
            double res = cost[m - 1][n - 1];
            return res;
        }
        
        private static final double calcualteCost(double new_point, double x, double y, double c) {

            double dist = 0;

            if (((x <= new_point) && (new_point <= y)) || ((y <= new_point) && (new_point <= x))) {
                dist = c;
            } else {
                dist = c + Math.min(Math.abs(new_point - x), Math.abs(new_point - y));
            }

            return dist;
        }
        
        private static double twed(double[] ta, double[] tb, double nu, double lambda) {

            int m = ta.length;
            int n = tb.length;
            int maxLength = Math.max(m, n);

            double dist, disti1, distj1;

            int r = ta.length; // this is just m?!
            int c = tb.length; // so is this, but surely it should actually
            // be n anyway

            int i, j;
            /*
             * allocations in c double **D = (double **)calloc(r+1,
             * sizeof(double*)); double *Di1 = (double *)calloc(r+1,
             * sizeof(double)); double *Dj1 = (double *)calloc(c+1,
             * sizeof(double)); for(i=0; i<=r; i++) { D[i]=(double
             * *)calloc(c+1, sizeof(double)); }
             */

            double[][]D = new double[maxLength + 1][maxLength + 1];
            double[]Di1 = new double[maxLength + 1];
            double[]Dj1 = new double[maxLength + 1];
            //		double[][] D = MemoryManager.getInstance().getDoubleMatrix(0);
            //		double[] Di1 = MemoryManager.getInstance().getDoubleArray(0);
            //		double[] Dj1 = MemoryManager.getInstance().getDoubleArray(1);

            // FPH adding initialisation given that using matrices as fields
            Di1[0] = 0.0;
            Dj1[0] = 0.0;
            // local costs initializations
            for (j = 1; j <= c; j++) {
                distj1 = 0;
                if (j > 1) {
                    // CHANGE AJB 8/1/16: Only use power of
                    // 2 for speed
                    distj1 += (tb[j - 2] - tb[j - 1]) * (tb[j - 2] - tb[j - 1]);
                    // OLD VERSION
                    // distj1+=Math.pow(Math.abs(tb[j-2][k]-tb[j-1][k]),degree);
                    // in c:
                    // distj1+=pow(fabs(tb[j-2][k]-tb[j-1][k]),degree);
                } else {
                    distj1 += tb[j - 1] * tb[j - 1];
                }
                // OLD distj1+=Math.pow(Math.abs(tb[j-1][k]),degree);
                Dj1[j] = (distj1);
            }

            for (i = 1; i <= r; i++) {
                disti1 = 0;
                if (i > 1) {
                    disti1 += (ta[i - 2] - ta[i - 1]) * (ta[i - 2] - ta[i - 1]);
                } // OLD
                // disti1+=Math.pow(Math.abs(ta[i-2][k]-ta[i-1][k]),degree);
                else {
                    disti1 += (ta[i - 1]) * (ta[i - 1]);
                }
                // OLD disti1+=Math.pow(Math.abs(ta[i-1][k]),degree);

                Di1[i] = (disti1);

                for (j = 1; j <= c; j++) {
                    dist = 0;
                    dist += (ta[i - 1] - tb[j - 1]) * (ta[i - 1] - tb[j - 1]);
                    // dist+=Math.pow(Math.abs(ta[i-1][k]-tb[j-1][k]),degree);
                    if (i > 1 && j > 1) {
                        dist += (ta[i - 2] - tb[j - 2]) * (ta[i - 2] - tb[j - 2]);
                    }
                    // dist+=Math.pow(Math.abs(ta[i-2][k]-tb[j-2][k]),degree);
                    D[i][j] = (dist);
                }
            } // for i

            // border of the cost matrix initialization
            D[0][0] = 0;
            for (i = 1; i <= r; i++) {
                D[i][0] = D[i - 1][0] + Di1[i];
            }
            for (j = 1; j <= c; j++) {
                D[0][j] = D[0][j - 1] + Dj1[j];
            }

            double dmin, htrans, dist0;

            for (i = 1; i <= r; i++) {
                for (j = 1; j <= c; j++) {
                    htrans = Math.abs(i- j);
                    if (j > 1 && i > 1) {
                        htrans += Math.abs((i-1) - (j-1));
                    }
                    dist0 = D[i - 1][j - 1] + nu * htrans + D[i][j];
                    dmin = dist0;
                    if (i > 1) {
                        htrans = 1;
                    } else {
                        htrans = i;
                    }
                    dist = Di1[i] + D[i - 1][j] + lambda + nu * htrans;
                    if (dmin > dist) {
                        dmin = dist;
                    }
                    if (j > 1) {
                        htrans = 1;
                    } else {
                        htrans = j;
                    }
                    dist = Dj1[j] + D[i][j - 1] + lambda + nu * htrans;
                    if (dmin > dist) {
                        dmin = dist;
                    }
                    D[i][j] = dmin;
                }
            }

            dist = D[r][c];
            return dist;
        }

        private static double[] initWeights(int seriesLength, double g) {
            
            double[] weightVector = new double[seriesLength];
            double halfLength = (double) seriesLength / 2;

            for (int i = 0; i < seriesLength; i++) {
                weightVector[i] = 1d / (1 + Math.exp(-g * (i - halfLength)));
            }
            
            return weightVector;
        }
        
        public static double wdtw(double[] first, double[] second, double g) {
            double[] weightVector = initWeights(first.length, g);

            double[] prevRow = new double[second.length];
            double[] curRow = new double[second.length];
            double second0 = second[0];
            double thisDiff;
            double prevVal = 0.0;

            // put the first row into prevRow to save swapping before moving to the second row

            {	double first0 = first[0];

                // first value
                thisDiff = first0 - second0;
                prevVal = prevRow[0] = weightVector[0] * (thisDiff * thisDiff);

                // top row
                for (int j = 1; j < second.length; j++) {
                    thisDiff = first0 - second[j];
                    prevVal = prevRow[j] = prevVal + weightVector[j] * (thisDiff * thisDiff);
                }
            }

            double minDistance;
            double firsti = first[1];

            // second row is a special case because path can't go through prevRow[j]
            thisDiff = firsti - second0;
            prevVal = curRow[0] = prevRow[0] + weightVector[1] * (thisDiff * thisDiff);

            for (int j = 1; j < second.length; j++) {
                // calculate distances
                minDistance = Math.min(prevVal, prevRow[j - 1]);
                thisDiff = firsti - second[j];
                prevVal = curRow[j] = minDistance + weightVector[j-1] * (thisDiff * thisDiff);
            }

            // warp rest
            for (int i = 2; i < first.length; i++) {
                // make the old current row into the current previous row and set current row to use the old prev row
                double [] tmp = curRow;
                curRow = prevRow;
                prevRow = tmp;
                firsti = first[i];

                thisDiff = firsti - second0;
                prevVal = curRow[0] = prevRow[0] + weightVector[i] * (thisDiff * thisDiff);

                for (int j = 1; j < second.length; j++) {
                    // calculate distances
                    minDistance = min(prevVal, prevRow[j], prevRow[j - 1]);
                    thisDiff = firsti - second[j];
                    prevVal = curRow[j] = minDistance + weightVector[Math.abs(i - j)] * (thisDiff * thisDiff);
                }

            }

            double res = prevVal;
            return res;
        }
        
        public static double wddtw(double[] a, double[] b, double g) {
            return wdtw(getDeriv(a), getDeriv(b), g);
        }
    }
}
