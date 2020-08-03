package tsml.classifiers.distance_based;

import experiments.data.DatasetLoading;
import org.apache.commons.lang3.ArrayUtils;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.transformers.*;
import utilities.generic_storage.Pair;
import utilities.samplers.RandomStratifiedSampler;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/*
 * This class extends on the idea of ShapeDTW_1NN by transforming
 * a test instance into ShapeDTW distance features (note that it uses
 * the default shape descriptor which is 'raw'). It calculates the test
 * instance's distance from a reference set and these become the features
 * for the instance. This transformed instance is then fed into a
 * SVM classifier. The same task is done on the training set.
 */
public class ShapeDTW_SVM extends EnhancedAbstractClassifier {

    // Defines the split from training to reference, 0.5 indicates
    // 0.5 train, 0.5 reference. 0.4 indicates 0.4 train, 0.6 reference
    // and so on.
    private double trainRefSplit = 1.0;
    private Subsequences subsequenceTransformer;
    //The transformer used to produce the shape dtw features.
    private ShapeDTWFeatures sdtwFeats;
    //The stratified sampler
    private RandomStratifiedSampler sampler;
    private int subsequenceLength;
    private SMO svmClassifier;
    public enum KernelType {PolyNomial,RBF};
    private KernelType kernelType = KernelType.PolyNomial;
    private Instances compoundDataset;
    private DimensionIndependentTransformer d1 = new DimensionIndependentTransformer(new DWT());
    private DimensionIndependentTransformer d2 = new DimensionIndependentTransformer(new Slope(5));

    public ShapeDTW_SVM() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceTransformer = new Subsequences();
        //Use a polynomial kernel (default exponent is 1, a linear kernel)
        this.svmClassifier = new SMO();
        this.sampler = new RandomStratifiedSampler();
    }

    public ShapeDTW_SVM(int subsequenceLength,KernelType k) {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceTransformer = new Subsequences(subsequenceLength);
        this.svmClassifier = new SMO();
        this.sampler = new RandomStratifiedSampler();
        this.kernelType = k;
    }
    /* Getters and Setters */
    public double getTrainRefSplit() { return trainRefSplit; }
    public void setTrainRefSplit(double trainRefSplit) {this.trainRefSplit = trainRefSplit;}
    public KernelType getKernelType() {return kernelType; }
    public void setKernelType(KernelType k) {this.kernelType = k;}

    @Override
    public void buildClassifier(Instances trainInsts) throws Exception {
        // Check the data
        this.getCapabilities().testWithFail(trainInsts);
        // Record the build time.
        long buildTime = System.nanoTime();
        // Convert the data into an appropriate form (ShapeDTW features)
        Instances trainingData = preprocessData(trainInsts);
        // Tune the SVM
        int numFolds = 10;
        tuneSVMClassifier(trainingData,numFolds);
        // Store the timing results.
        buildTime = System.nanoTime() - buildTime ;
        this.trainResults.setBuildTime(buildTime);
        this.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
    }

    /**
     * Private method for tuning the SVM classifier. The most important
     * parameters is the degree of the polynomial kernel (linear, quadratic
     * or cubic) and the slack variable C (10^-5,...,10^5). Performs a
     * 10-fold cross-validation. Parameters are the training data, the number of folds
     * and the number of iterations (as its a Random Search).
     */
    private void tuneSVMClassifier(Instances trainData, int numFolds) throws Exception {
        Instances [] folds = createFolds(numFolds,trainData);
        Double [] paramArray;
        Double [] cs = new Double[]{0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0};
        if(this.kernelType == KernelType.PolyNomial) {
            //The selected degrees to try
            paramArray = new Double[]{1.0,2.0,3.0};
        } else {
            //a RBF kernel
            //selected Gamma values
            paramArray = new Double[]{0.005,0.01,0.05,0.1,0.5,1.0,5.0,10.0};
        }
        int numIters = paramArray.length*cs.length;
        ArrayList<Pair<Double,Double>> paramsToTry = createParamsToTry(paramArray,cs,numIters);
        //try the selected parameter values
        double [] accuracies = collectResults(folds,paramsToTry);
        //Choose the parameter with the best accuracy (ties are settled randomly)
        ArrayList<Integer> bestIndex = new ArrayList<>();
        //Gather the bestAccuracies
        double bestAccuracy = -Double.MAX_VALUE;
        for(int i=0;i<accuracies.length;i++) {
            if(accuracies[i] > bestAccuracy) {
                bestAccuracy = accuracies[i];
                bestIndex = new ArrayList<>();
                bestIndex.add(i);
            } else if(accuracies[i] == bestAccuracy) {
                bestIndex.add(i);
            }
        }
        //Choose the configuration
        Pair<Double,Double> chosenParamValue;
        if(bestIndex.size() == 1) {
            chosenParamValue = paramsToTry.get(bestIndex.get(0));
        } else {
            Random rnd = new Random();
            int index = rnd.nextInt(bestIndex.size());
            chosenParamValue = paramsToTry.get(bestIndex.get(index));
        }
        //Set up the SVM as the chosen configuration
        if(kernelType == KernelType.PolyNomial) {
            PolyKernel p = new PolyKernel();
            p.setExponent(chosenParamValue.var1);
            this.svmClassifier.setKernel(p);
            this.svmClassifier.setC(chosenParamValue.var2);
            System.out.println("Chosen Exponent - " + chosenParamValue.var1);
            System.out.println("Chosen C Value - " + chosenParamValue.var2);
        } else {
            RBFKernel r = new RBFKernel();
            r.setGamma(chosenParamValue.var1);
            this.svmClassifier.setKernel(r);
            this.svmClassifier.setC(chosenParamValue.var2);
            System.out.println("Chosen Gamma Value - " + chosenParamValue.var1);
            System.out.println("Chosen C Value - " + chosenParamValue.var2);
        }
        this.svmClassifier.buildClassifier(trainData);
    }

    /**
     * Private method for collecting the accuracy of the SVM using the selected parameter values.
     *
     * @param folds
     * @param paramsToTry
     * @return
     */
    private double [] collectResults(Instances [] folds, ArrayList<Pair<Double,Double>> paramsToTry) throws Exception{
        double [] paramsAccuracies = new double[paramsToTry.size()];
        int total = folds.length*folds[0].numInstances();
        int numFolds = folds.length;
        //Go through every parameter value
        for(int i=0;i<paramsToTry.size();i++) {
            System.out.println("Param number - " + i);
            //Perform the n-fold cross-validation
            //record the number it predicted correctly.
            int numCorrect = 0;
            //For each fold
            for(int j=0;j<folds.length;j++) {
                List<Integer> trainFolds = IntStream.rangeClosed(0,numFolds-1).boxed().collect(Collectors.toList());
                //testing fold
                Instances test = folds[j];
                trainFolds.remove(j);
                //training folds
                Instances train = createTrainFolds(folds,trainFolds);
                //set up the SVM
                if(kernelType == KernelType.PolyNomial) {
                    PolyKernel p = new PolyKernel();
                    p.setExponent(paramsToTry.get(i).var1);
                    this.svmClassifier.setKernel(p);
                    this.svmClassifier.setC(paramsToTry.get(i).var2);
                    this.svmClassifier.buildClassifier(train);
                } else {
                    RBFKernel r = new RBFKernel();
                    r.setGamma(paramsToTry.get(i).var1);
                    this.svmClassifier.setKernel(r);
                    this.svmClassifier.setC(paramsToTry.get(i).var2);
                    this.svmClassifier.buildClassifier(train);
                }
                //test the SVM under the current configuration
                for(int k=0;k<test.numInstances();k++) {
                    double predictedClassValue = this.svmClassifier.classifyInstance(test.get(k));
                    if(predictedClassValue == test.get(k).classValue()) {
                        numCorrect++;
                    }
                }
            }
            paramsAccuracies[i] = (double) numCorrect / (double) total;
        }
        return paramsAccuracies;
    }

    /**
     * Private function for combining multiple instances into one.
     *
     * @return
     */
    private Instances createTrainFolds(Instances [] listOfFolds, List<Integer> requiredFolds) {
        Instances combinedInsts = new Instances(listOfFolds[0],
                                        listOfFolds[0].numInstances()*requiredFolds.size());

        for(int i=0;i<requiredFolds.size();i++) {
            Instances fold = listOfFolds[requiredFolds.get(i)];
            for(int j=0;j<fold.numInstances();j++) {
                combinedInsts.add(fold.get(j));
            }
        }
        return combinedInsts;
    }

    /**
     * Private function for choosing the combination of parameters to try. First
     * value is the chosen degree, second is the chosen C value.
     *
     * @param degrees
     * @param cVals
     * @param numIts
     * @return
     */
    private ArrayList<Pair<Double,Double>> createParamsToTry(Double [] degrees, Double [] cVals, int numIts) {
        Random rnd = new Random();
        ArrayList<Pair<Double,Double>> chosenParams = new ArrayList<>();
        while(chosenParams.size() != numIts) {
            Double chosenDegree = degrees[rnd.nextInt(degrees.length)];
            Double chosenC = cVals[rnd.nextInt(cVals.length)];
            Pair<Double,Double> chosenParam = new Pair<>(chosenDegree,chosenC);
            if(chosenParams.contains(chosenParam)) {
                //If this parameter pair has already been chosen, choose a different one.
                continue;
            } else {
                chosenParams.add(chosenParam);
            }
        }
        return chosenParams;
    }

    /**
     * Private method to create the folds used to tune the SVM classifier.
     *
     * @param numFolds
     * @return
     */
    private Instances [] createFolds(int numFolds,Instances trainData) {
        int foldSize = (int) Math.floor((double)trainData.numInstances()/ (double) numFolds);
        Instances [] folds = new Instances [numFolds];
        for(int i=0;i<numFolds;i++) {
            folds[i] = new Instances(trainData,foldSize);
            for(int j=0+foldSize*i;j<foldSize + foldSize*i;j++) {
                folds[i].add(trainData.get(j));
            }
        }
        return folds;
    }

    /**
     * Private method for performing the subsequence extraction on an instance and creating the ShapeDTW features.
     *
     * @param testInsts
     * @return just the training set.
     */
    private Instance preprocessData(Instance testInsts) {
        // Transform the trainInsts into subsequences
        Instance subsequences = this.subsequenceTransformer.transform(testInsts);
        // Then, transform into compound shape descs (used DWT and Slope)
        Instance s1 = this.d1.transform(subsequences);
        Instance s2 = this.d2.transform(subsequences);
        Instance res = combineInstances(s1,s2);
        return this.sdtwFeats.transform(res);
    }


    /**
     * Private method for performing the subsequence extraction on an instance,
     * performing a stratfied sample and creating the ShapeDTW features.
     *
     * @param trainInsts
     * @return just the training set.
     */
    private Instances preprocessData(Instances trainInsts) {
        // Transform the trainInsts into subsequences
        Instances subsequences = this.subsequenceTransformer.transform(trainInsts);
        // Then, transform into compound shape descs (used DWT and Slope)
        // this was found to be the most powerful one
        Instances s1 = this.d1.transform(subsequences);
        Instances s2 = this.d2.transform(subsequences);
        Instances res = combineInstances(s1,s2);
        // Create the shapeDTW features on the training set
        this.sdtwFeats = new ShapeDTWFeatures(res);
        return this.sdtwFeats.transform(res);
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
    public double classifyInstance(Instance testInst) throws Exception {
        Instance testData = preprocessData(testInst);
        return this.svmClassifier.classifyInstance(testData);
    }

    @Override
    public double [] distributionForInstance(Instance testInst) throws Exception {
        Instance testData = preprocessData(testInst);
        return this.svmClassifier.distributionForInstance(testData);
    }

    /**
     * Testing method for this class.
     *
     * @param args - the command line arguments.
     */
    public static void main(String[] args) throws Exception {
        Instances [] data = DatasetLoading.sampleBeef(0);
        ShapeDTW_SVM s = new ShapeDTW_SVM();
        s.setKernelType(KernelType.PolyNomial);
        System.out.println(calculateAccuracy(s,data));
        ShapeDTW_1NN s2 = new ShapeDTW_1NN();
        System.out.println(calculateAccuracy(s2,data));
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
     * Function to calculate accuracy.
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
}
