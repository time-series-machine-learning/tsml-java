package tsml.classifiers.distance_based;

import org.apache.commons.lang3.ArrayUtils;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.multivariate.MultivariateAbstractClassifier;
import tsml.transformers.DWT;
import tsml.transformers.HOG1D;
import tsml.transformers.Slope;
import tsml.transformers.Subsequences;
import tsml.transformers.*;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import java.util.ArrayList;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;

/**
 * The ShapeDTW classifier works by initially extracting a set of subsequences
 * describing local neighbourhoods around each data point in a time series.
 * These subsequences are then passed into a shape descriptor function that
 * transforms these local neighbourhoods into a new representation. This
 * new representation is then sent into DTW with 1-NN (DTW_D).
 *
 * @author Vincent Nicholson
 *
 */
public class ShapeDTW_1NN extends EnhancedAbstractClassifier {

    // hyper-parameters
    private int subsequenceLength;
    // if shapeDescriptor is null, then its the 'raw' shape descriptor.
    // Supported Transformers are the following:
    // null - raw
    // PAA
    // DWT
    // Derivative
    // Slope
    // HOG1D
    private Transformer shapeDescriptor;
    //The second shape descriptor is for performing another transformation and concatenating the results together.
    private Transformer secondShapeDescriptor = null;
    //Authors also propose a weighting factor which is a value to multiply the output of the second transformation
    //by. This produces an output in the form compound = (ShapeDescriptor,weightingFactor*secondShapeDescriptor).
    private double weightingFactor = 1.0;
    // Transformer for extracting the neighbourhoods
    private Subsequences subsequenceTransformer;
    // NN_DTW_Subsequences for performing classification on the training data
    private NN_DTW_Subsequences nnDtwSubsequences;
    //The Dimension independent transformers
    private DimensionIndependentTransformer d1;
    private DimensionIndependentTransformer d2;
    // Another method proposed is to combine the results of two shapeDescriptors together, if this is set to
    // true, then the results of shapeDescriptor and secondShapeDescriptor are concatenated together.
    private boolean useSecondShapeDescriptor = false;
    private final Transformer [] VALID_TRANSFORMERS = new Transformer[] {new PAA(), new DWT(),
                                                                         new Derivative(), new Slope(),
                                                                         new HOG1D()};
    // For storing the dataset when creating the compound shape descriptors.
    private Instances compoundDataset;

    /**
     * Private constructor with settings:
     * subsequenceLength = 30
     * shapeDescriptorFunction = null (so 'raw' is used)
     */
    public ShapeDTW_1NN() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceLength = 30;
        this.shapeDescriptor = null;
        this.subsequenceTransformer = new Subsequences(subsequenceLength);
        this.nnDtwSubsequences = new NN_DTW_Subsequences();
    }

    public ShapeDTW_1NN(int subsequenceLength,Transformer shapeDescriptor,boolean useSecondShapeDescriptor,
                       Transformer secondShapeDescriptor) {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceLength = subsequenceLength;
        this.shapeDescriptor = shapeDescriptor;
        this.subsequenceTransformer = new Subsequences(subsequenceLength);
        this.nnDtwSubsequences = new NN_DTW_Subsequences();
        this.secondShapeDescriptor = secondShapeDescriptor;
        this.useSecondShapeDescriptor = useSecondShapeDescriptor;
    }

    public int getSubsequenceLength() {
        return subsequenceLength;
    }

    public Transformer getShapeDescriptors() {
        return shapeDescriptor;
    }

    public boolean isUsingSecondShapeDescriptor() {return useSecondShapeDescriptor; }

    public Transformer getSecondShapeDescriptor() {return secondShapeDescriptor;}

    public double getWeightingFactor() {return weightingFactor; }

    public void setSubsequenceLength(int subsequenceLength) {
        this.subsequenceLength = subsequenceLength;
    }

    public void setShapeDescriptors(Transformer shapeDescriptors) {
        this.shapeDescriptor = shapeDescriptors;
    }

    public void setIsUsingSecondShapeDescriptor(boolean flag) {this.useSecondShapeDescriptor = flag; }

    public void setSecondShapeDescriptor(Transformer t) {this.secondShapeDescriptor = t; }

    public void setWeightingFactor(double newWeightingFactor) {this.weightingFactor = newWeightingFactor;}

    /**
     * Private method for performing the subsequence extraction on a set of instances as
     * well as the shape descriptor function for training (if not null).
     *
     * @param data
     * @return
     */
    private Instances preprocessData(Instances data) {
        Instances transformedData = this.subsequenceTransformer.transform(data);
        Instances shapeDesc1;
        Instances shapeDesc2 = null;
        //If shape descriptor is null aka 'raw', use the subsequences.
        if (this.shapeDescriptor == null) {
            shapeDesc1 = new Instances(transformedData);
        } else {
            this.d1 = new DimensionIndependentTransformer(this.shapeDescriptor);
            shapeDesc1 = this.d1.transform(transformedData);
        }
        //Test if a second shape descriptor is required
        if(useSecondShapeDescriptor) {
            if(this.secondShapeDescriptor == null) {
                shapeDesc2 = new Instances(transformedData);
            } else {
                this.d2 = new DimensionIndependentTransformer(this.secondShapeDescriptor);
                shapeDesc2 = this.d2.transform(transformedData);
            }
        }
        Instances combinedInsts = combineInstances(shapeDesc1,shapeDesc2);
        return combinedInsts;
    }

    /**
     * Private method for performing the subsequence extraction on an instance as
     * well as the shape descriptor function for testing (if not null).
     *
     * @param data
     * @return
     */
    private Instance preprocessData(Instance data) {
        Instance transformedData = this.subsequenceTransformer.transform(data);
        Instance shapeDesc1;
        Instance shapeDesc2 = null;
        //If shape descriptor is null aka 'raw', use the subsequences.
        if (this.shapeDescriptor == null) {
            shapeDesc1 = transformedData;
        } else {
            shapeDesc1 = this.d1.transform(transformedData);
        }
        //Test if a second shape descriptor is required
        if(useSecondShapeDescriptor) {
            if(this.secondShapeDescriptor == null) {
                shapeDesc2 = transformedData;
            } else {
                shapeDesc2 = this.d2.transform(transformedData);
            }
        }
        Instance combinedInsts = combineInstances(shapeDesc1,shapeDesc2);
        return combinedInsts;
    }

    /**
     * Private function for concatenating two shape descriptors together.
     *
     * @param shapeDesc1
     * @param shapeDesc2
     * @return
     */
    private Instances combineInstances(Instances shapeDesc1,Instances shapeDesc2) {
        if(shapeDesc2 == null) {
            return shapeDesc1;
        }
        //Create the header for the new data to be stored in.
        Instances compoundHeader = createCompoundHeader(shapeDesc1,shapeDesc2);
        for(int i=0;i<shapeDesc1.numInstances();i++) {
            Instances relationHeader = new Instances(compoundHeader.attribute(0).relation());
            DenseInstance newInst = new DenseInstance(2);
            newInst.setDataset(compoundHeader);
            //Combine all the dimensions together to create the relation
            Instances relation = createRelationalData(shapeDesc1.get(i), shapeDesc2.get(i), relationHeader);
            //Add relation to the first value of newInst
            int index = newInst.attribute(0).addRelation(relation);
            newInst.setValue(0, index);
            //Add the class value.
            newInst.setValue(1, shapeDesc1.get(i).classValue());
            compoundHeader.add(newInst);
        }
        compoundHeader.setClassIndex(1);
        this.compoundDataset = compoundHeader;
        return compoundHeader;
    }

    /**
     * Private function for creating the header for the compound shape descriptor data.
     *
     * @param shapeDesc1 - Instances of the first shape descriptor.
     * @param shapeDesc2 - Instances of the second shape descriptor.
     * @return
     */
    private Instances createCompoundHeader(Instances shapeDesc1,Instances shapeDesc2) {
        // Create the Instances object
        ArrayList<Attribute> atts = new ArrayList<>();
        //Create the relational attribute
        ArrayList<Attribute> relationalAtts = new ArrayList<>();
        int numAttributes = shapeDesc1.attribute(0).relation().numAttributes() +
                shapeDesc2.attribute(0).relation().numAttributes();
        // Add the original elements
        for (int i = 0; i < numAttributes; i++)
            relationalAtts.add(new Attribute("Compound_element_" + i));
        // Create the relational table
        Instances relationTable = new Instances("Compound_Elements", relationalAtts, shapeDesc1.numInstances());
        // Create the attribute from the relational table
        atts.add(new Attribute("relationalAtt", relationTable));
        // Add the class attribute
        atts.add(shapeDesc1.classAttribute());
        Instances compoundShapeDesc = new Instances("Compound_Elements",atts,shapeDesc1.numInstances());
        return compoundShapeDesc;
    }

    /**
     * Private function for creating the relation along each dimension within
     * inst1 and inst2.
     *
     * @param inst1
     * @param inst2
     * @return
     */
    private Instances createRelationalData(Instance inst1, Instance inst2, Instances header) {
        Instances rel1 = inst1.relationalValue(0);
        Instances rel2 = inst2.relationalValue(0);

        //  Iterate over each dimension
        for(int i=0;i<rel1.numInstances();i++) {
            double [] dim1 = rel1.get(i).toDoubleArray();
            double [] dim2 = rel2.get(i).toDoubleArray();
            //multiply dim2 by a weighting factor
            for(int j=0;j<dim2.length;j++) {
                dim2[j] = dim2[j]*this.weightingFactor;
            }
            double [] both = ArrayUtils.addAll(dim1,dim2);
            //Create the new Instance
            DenseInstance newInst = new DenseInstance(both.length);
            for(int j=0;j<both.length;j++) {
                newInst.setValue(j,both[j]);
            }
            header.add(newInst);
        }
        return header;
    }

    /**
     * Private function for concatenating two shape descriptors together.
     *
     * @param shapeDesc1
     * @param shapeDesc2
     * @return
     */
    private Instance combineInstances(Instance shapeDesc1, Instance shapeDesc2) {
        if(shapeDesc2 == null) {
            return shapeDesc1;
        }
        Instance combinedInst = new DenseInstance(2);
        //Create the relational table
        ArrayList<Attribute> relationalAtts = new ArrayList<>();
        int numAttributes = shapeDesc1.attribute(0).relation().numAttributes() +
                shapeDesc2.attribute(0).relation().numAttributes();
        // Add the original elements
        for (int i = 0; i < numAttributes; i++)
            relationalAtts.add(new Attribute("Compound_element_" + i));
        // Create the relational table
        Instances relationTable = new Instances("Compound_Elements", relationalAtts,
                                                shapeDesc1.attribute(0).relation().numInstances());
        Instances relation = createRelationalData(shapeDesc1,shapeDesc2,relationTable);
        combinedInst.setDataset(this.compoundDataset);
        int index = combinedInst.attribute(0).addRelation(relation);
        combinedInst.setValue(0, index);
        //Add the class value.
        combinedInst.setValue(1, shapeDesc1.classValue());
        return combinedInst;
    }

    @Override
    public void buildClassifier(Instances trainInst) throws Exception {
        // Check the given parameters
        this.checkParameters();
        // Check the data
        this.getCapabilities().testWithFail(trainInst);
        // Record the build time.
        long buildTime = System.nanoTime();
        // Train the classifier
        Instances transformedData = this.preprocessData(trainInst);
        this.nnDtwSubsequences.buildClassifier(transformedData);
        // Store the timing results.
        buildTime = System.nanoTime() - buildTime ;
        this.trainResults.setBuildTime(buildTime);
        this.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
    }

    @Override
    public double [] distributionForInstance(Instance testInst) throws Exception {
        Instance transformedData = this.preprocessData(testInst);
        return this.nnDtwSubsequences.distributionForInstance(transformedData);
    }

    @Override
    public double classifyInstance(Instance testInst) throws Exception {
        Instance transformedData = this.preprocessData(testInst);
        return this.nnDtwSubsequences.classifyInstance(transformedData);
    }

    /**
     * Private method for checking the parameters inputted into ShapeDTW.
     *
     */
    private void checkParameters() {
        if(this.subsequenceLength < 1) {
            throw new IllegalArgumentException("subsequenceLength cannot be less than 1.");
        }
        //Check the shapeDescriptor function is the correct type.
        boolean found = false;
        for(Transformer x: this.VALID_TRANSFORMERS) {
            if(this.shapeDescriptor == null) {
                found = true;
                break;
            }
            if(this.shapeDescriptor.getClass().equals(x.getClass())) {
                found = true;
                break;
            }
        }
        if(!found) {
            throw new IllegalArgumentException("Invalid transformer type for shapeDescriptor.");
        }
        //Check the secondShapeDescriptor function is the correct type.
        found = false;
        for(Transformer x: this.VALID_TRANSFORMERS) {
            if(this.secondShapeDescriptor == null) {
                found = true;
                break;
            }
            if(this.secondShapeDescriptor.getClass().equals(x.getClass())) {
                found = true;
                break;
            }
        }
        if(!found) {
            throw new IllegalArgumentException("Invalid transformer type for shapeDescriptor.");
        }
    }

    /**
     * Main method for testing.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        Instances trainData = createTrainData();
        Instances testData = createTestData();

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        String datasetName = "InsectEPGSmallTrain";
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Univariate2018_arff\\" + datasetName + "\\" + datasetName + "_TRAIN.arff");
        Instances train = source.getDataSet();
        if (train.classIndex() == -1) {
            train.setClassIndex(train.numAttributes() - 1);
        }
        source = new ConverterUtils.DataSource("C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Univariate2018_arff\\" + datasetName + "\\" + datasetName + "_TEST.arff");
        Instances test = source.getDataSet();
        if (test.classIndex() == -1) {
            test.setClassIndex(test.numAttributes() - 1);
        }
        Instances [] data = new Instances[] {train,test};

        // test bad subsequence length
        // this has to be greater than 0.
        /*int [] badSubsequences = new int [] {0,-1,-999999999};
        for(int i=0;i<badSubsequences.length;i++) {
            try{
                ShapeDTW_1NN s = new ShapeDTW_1NN(badSubsequences[i],null,false,null);
                s.buildClassifier(trainData);
                System.out.println("Test failed.");
            } catch(IllegalArgumentException e) {
                System.out.println("Test passed.");
            }
        }
        // test good subsequence length
        int [] goodSubsequences = new int [] {1,5,999};
        for(int i=0;i<goodSubsequences.length;i++) {
            try{
                ShapeDTW_1NN s = new ShapeDTW_1NN(goodSubsequences[i],null,false,null);
                s.buildClassifier(trainData);
                System.out.println("Test passed.");
            } catch(IllegalArgumentException e) {
                System.out.println("Test failed.");
            }
        }
        // test bad transformer
        // Can only be null, PAA, DWT, Derivative, Slope or HOG1D
        Transformer [] badTransformer = new Transformer [] {new PCA(),new Cosine(),new FFT()};
        for(int i=0;i<badTransformer.length;i++) {
            try{
                ShapeDTW_1NN s = new ShapeDTW_1NN(30,badTransformer[i],false,null);
                s.buildClassifier(trainData);
                System.out.println("Test failed.");
            } catch(IllegalArgumentException e) {
                System.out.println("Test passed.");
            }
        }
        // test good transformer
        Transformer [] badTransformers = new Transformer[] {new PAA(), null, new HOG1D()};
        for(int i=0;i<badTransformers.length;i++) {
            try{
                ShapeDTW_1NN s = new ShapeDTW_1NN(30,badTransformers[i],false,null);
                s.buildClassifier(trainData);
                System.out.println("Test passed.");
            } catch(IllegalArgumentException e) {
                System.out.println("Test failed.");
            }
        }
        // test classification functionality
        //Instances [] data = new Instances [] {trainData,testData};
        Transformer [] allTrans = new Transformer [] {null,new PAA(), new DWT(), new Derivative(), new Slope(),
                                                      new HOG1D()};
        for(Transformer t:allTrans) {
            ShapeDTW_1NN s = new ShapeDTW_1NN(30,t,false,null);
            System.out.println(calculateAccuracy(s,data));
            System.out.println("Test passed.");
        }*/
        // Test compound shape descriptor functionality.
        //ShapeDTW_1NN s = new ShapeDTW_1NN(1,null,false,new Slope());
        //System.out.println(calculateAccuracy(s,data));

        ShapeDTW_1NN s = new ShapeDTW_1NN(30,null,false,new Slope());
        System.out.println(calculateAccuracy(s,data));
        System.out.println("Test passed.");
    }

    /**
     * Function to calculate accuracy purely for testing the functionality of ShapeDTW_1NN.
     *
     * @param s
     * @param data
     * @return
     * @throws Exception
     */
    private static double calculateAccuracy(AbstractClassifier s, Instances [] data) throws Exception {
        Instances train = data[0];
        Instances test = data[1];

        s.buildClassifier(train);
        int correct = 0;
        for(int i=0;i<test.numInstances();i++) {
            double predict = s.classifyInstance(test.get(i));
            if(predict == test.get(i).classValue()) {
                correct++;
            }
        }
        return (double) correct/(double) test.numInstances();
    }

    /**
     * Function to create train data for testing purposes.
     *
     * @return
     */
    private static Instances createTrainData() {
        //Create the attributes
        ArrayList<Attribute> atts = new ArrayList<>();
        for(int i=0;i<5;i++) {
            atts.add(new Attribute("test_" + i));
        }
        //Create the class values
        ArrayList<String> classes = new ArrayList<>();
        classes.add("1");
        classes.add("0");
        atts.add(new Attribute("class",classes));
        Instances newInsts = new Instances("Test_dataset",atts,5);
        newInsts.setClassIndex(newInsts.numAttributes()-1);

        //create the test data
        double [] test = new double [] {1,2,3,4,5};
        createInst(test,"1",newInsts);
        test = new double [] {1,1,2,3,4};
        createInst(test,"1",newInsts);
        test = new double [] {2,2,2,3,4};
        createInst(test,"0",newInsts);
        test = new double [] {2,3,4,5,6};
        createInst(test,"0",newInsts);
        test = new double [] {0,1,1,1,2};
        createInst(test,"1",newInsts);
        return newInsts;
    }

    /**
     * Function to create test data for testing purposes.
     *
     * @return
     */
    private static Instances createTestData() {
        //Create the attributes
        ArrayList<Attribute> atts = new ArrayList<>();
        for(int i=0;i<5;i++) {
            atts.add(new Attribute("test_" + i));
        }
        //Create the class values
        ArrayList<String> classes = new ArrayList<>();
        classes.add("1");
        classes.add("0");
        atts.add(new Attribute("class",classes));
        Instances newInsts = new Instances("Test_dataset",atts,5);
        newInsts.setClassIndex(newInsts.numAttributes()-1);

        //create the test data
        double [] test = new double [] {2,4,6,8,10};
        createInst(test,"1",newInsts);
        test = new double [] {1,1,2,3,4};
        createInst(test,"1",newInsts);
        test = new double [] {0,1,1,1,2};
        createInst(test,"1",newInsts);
        return newInsts;
    }

    /**
     * private function for creating an instance from a double array. Used
     * for testing purposes.
     *
     * @param arr
     * @return
     */
    private static void createInst(double [] arr,String classValue, Instances dataset) {
        Instance inst = new DenseInstance(arr.length+1);
        for(int i=0;i<arr.length;i++) {
            inst.setValue(i,arr[i]);
        }
        inst.setDataset(dataset);
        inst.setClassValue(classValue);
        dataset.add(inst);
    }

    /**
     * Inner class for calculating the DTW distance between subsequences used for the
     * classification stage (no warping window for this implementation).
     *
     * TODO: Make this not static once testing has been completed.
     */
    public static class NN_DTW_Subsequences extends MultivariateAbstractClassifier {

        private Instances train;

        public NN_DTW_Subsequences() {};

        /**
         * @param data set of instances serving as training data
         * @throws Exception
         */
        @Override
        public void buildClassifier(Instances data) throws Exception {
            this.train = data;
        }

        /**
         * Find the instance with the lowest distance.
         *
         * @param inst
         * @return
         */
        @Override
        public double classifyInstance(Instance inst) {
            //A TreeMap representing the distance from the test instance
            //and the training instance, and an arraylist of instances
            //representing the training instances with that distance.
            TreeMap<Double,ArrayList<Instance>> treeMap = new TreeMap<>();
            Double dist;
            ArrayList<Instance> insts;
            for(Instance i: train) {
                dist = calculateDistance(i,inst);
                // If distance has been found before
                if(treeMap.containsKey(dist)) {
                    insts = treeMap.get(dist);
                    insts.add(i);
                    treeMap.put(dist,insts);
                } else {
                    insts = new ArrayList<>();
                    insts.add(i);
                    treeMap.put(dist, insts);
                }
            }
            //return the instance with the lowest distance
            Double lowestDist = treeMap.firstKey();
            ArrayList<Instance> clostestInsts = treeMap.get(lowestDist);
            //If only one, return it.
            if(clostestInsts.size() == 1) {
                return clostestInsts.get(0).classValue();
            }
            //Else, choose a random one
            Random rnd = new Random(0);
            int index = rnd.nextInt(clostestInsts.size());
            return clostestInsts.get(index).classValue();
        }

        /**
         * Private function for calculating the distance between two instances
         * that are represented as a set of subsequences. It's static so that
         * it can be used elsewhere (also used in the ShapeDTWFeatures transformer).
         *
         * @param inst1
         * @param inst2
         * @return
         */
        public static double calculateDistance(Instance inst1, Instance inst2) {
            /*
             * This code has been adopted from the Slow_DTW_1NN class except for a few differences.
             */
            Instance [] inst1Arr = MultivariateInstanceTools.splitMultivariateInstance(inst1);
            Instance [] inst2Arr = MultivariateInstanceTools.splitMultivariateInstance(inst2);
            double [] [] a = MultivariateInstanceTools.convertMultiInstanceToArrays(inst1Arr);
            double [] [] b = MultivariateInstanceTools.convertMultiInstanceToArrays(inst2Arr);

            double minDist;
            int n=a.length;
            int m=b.length;
            double [] [] matrixD = new double[n][n];
            double windowSize = n;
            /*
            //Set all to max. This is necessary for the window but I dont need to do
            it all
            */
            for(int i=0;i<n;i++)
                for(int j=0;j<m;j++)
                    matrixD[i][j]=Double.MAX_VALUE;

            // Original DTW was a[0]-b[0]^2 (difference between points),
            // now its the euclidean distance between subsequences.
            matrixD[0][0]=calculateEuclideanDistance(a[0],b[0]);

            //Base cases for warping 0 to all with max interval	r
            //Warp a[0] onto all b[1]...b[r+1]
            for(int j=1;j<windowSize && j<n;j++)
                                              //also the case here
                matrixD[0][j]=matrixD[0][j-1]+calculateEuclideanDistance(a[0],b[j]);

            //	Warp b[0] onto all a[1]...a[r+1]
            for(int i=1;i<windowSize && i<n;i++)
                matrixD[i][0]=matrixD[i-1][0]+calculateEuclideanDistance(a[i],b[0]);
            //Warp the rest,
            for (int i=1;i<n;i++){
                for (int j = 1;j<m;j++){
                    //Find the min of matrixD[i][j-1],matrixD[i-1][j] and matrixD[i-1][j-1]
                    if (i < j + windowSize && j < i + windowSize) {
                        minDist=matrixD[i][j-1];
                        if(matrixD[i-1][j]<minDist)
                            minDist=matrixD[i-1][j];
                        if(matrixD[i-1][j-1]<minDist)
                            minDist=matrixD[i-1][j-1];
                        matrixD[i][j]=minDist+calculateEuclideanDistance(a[i],b[j]);
                    }
                }
            }
            //Find the minimum distance at the end points, within the warping window.
            return matrixD[n-1][m-1];
        }

        /**
         * Private function for calculating the euclidean distance between two subsequences.
         *
         * @param subsequence1
         * @param subsequence2
         * @return
         */
        private static double calculateEuclideanDistance(double [] subsequence1, double [] subsequence2) {
            double total = 0.0;
            for(int i=0;i<subsequence1.length;i++) {
                total += Math.pow(subsequence1[i] - subsequence2[i],2);
            }
            return total;
        }

        public static void main(String[] args) {
            try{
                NN_DTW_Subsequences d = new NN_DTW_Subsequences();
                ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:\\Users\\Vince\\Documents\\Dissertation Repositories\\test.arff");
                Instances data = source.getDataSet();
                if (data.classIndex() == -1) {
                    data.setClassIndex(data.numAttributes() - 1);
                }
                System.out.println(data.toString());
                System.out.println(d.calculateDistance(data.get(2),data.get(3)));
            } catch(Exception e) {

            }
        }
    }
}
