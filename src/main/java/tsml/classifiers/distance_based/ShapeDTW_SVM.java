package tsml.classifiers.distance_based;

import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.transformers.ShapeDTWFeatures;
import tsml.transformers.Subsequences;
import utilities.generic_storage.Pair;
import utilities.samplers.RandomStratifiedSampler;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
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

    private Subsequences subsequenceTransformer;
    //The transformer used to produce the shape dtw features.
    private ShapeDTWFeatures sdtwFeats;
    //The stratified sampler
    private RandomStratifiedSampler sampler;
    private int subsequenceLength;
    private SMO svmClassifier;
    public enum KernelType {PolyNomial,RBF};
    private KernelType kernelType = KernelType.PolyNomial;

    public ShapeDTW_SVM() {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceTransformer = new Subsequences();
        //Use a polynomial kernel (default exponent is 1, a linear kernel)
        this.svmClassifier = new SMO();
        this.sampler = new RandomStratifiedSampler();
    }

    public ShapeDTW_SVM(int subsequenceLength, KernelType k) {
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        this.subsequenceTransformer = new Subsequences(subsequenceLength);
        this.svmClassifier = new SMO();
        this.sampler = new RandomStratifiedSampler();
        this.kernelType = k;
    }
    /* Getters and Setters */
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
        // Tune the exponent (just values 1,2 and 3 and C values 10^-5 to 10^5).
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
     * Private method for performing the subsequence extraction on an instance,
     * transforming it using DWT and creating the ShapeDTW features.
     *
     * @param trainInsts
     */
    private Instances preprocessData(Instances trainInsts) {
        // Transform the trainInsts into subsequences
        Instances subsequences = this.subsequenceTransformer.transform(trainInsts);
        // Create the shapeDTW features on the training set
        this.sdtwFeats = new ShapeDTWFeatures(subsequences);
        return this.sdtwFeats.transform(subsequences);
    }

    /**
     * Private method for performing the subsequence extraction on an instance,
     * transforming it using DWT and creating the ShapeDTW features.
     *
     * @param trainInsts
     * @return just the training set (the reference set is stored in ShapeDtwFeatures).
     */
    private Instance preprocessData(Instance trainInsts) {
        // Transform the trainInsts into subsequences
        Instance subsequences = this.subsequenceTransformer.transform(trainInsts);
        // Create the shapeDTW features
        return this.sdtwFeats.transform(subsequences);
    }

    @Override
    public double classifyInstance(Instance testInst) throws Exception {
        Instance transformedInst = this.preprocessData(testInst);
        return this.svmClassifier.classifyInstance(transformedInst);
    }

    @Override
    public double [] distributionForInstance(Instance testInst) throws Exception {
        Instance transformedInst = this.preprocessData(testInst);
        return this.svmClassifier.distributionForInstance(transformedInst);
    }

    /**
     * Testing method for this class.
     *
     * @param args - the command line arguments.
     */
    public static void main(String[] args) throws Exception {
        Instances [] data = DatasetLoading.sampleBeef(0);
        ShapeDTW_SVM s = new ShapeDTW_SVM();
        s.setKernelType(KernelType.RBF);
        System.out.println(calculateAccuracy(s,data));
        ShapeDTW_1NN s2 = new ShapeDTW_1NN();
        System.out.println(calculateAccuracy(s2,data));
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
