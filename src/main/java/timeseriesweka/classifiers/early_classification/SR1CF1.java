package timeseriesweka.classifiers.early_classification;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;

import static experiments.ExperimentsEarlyClassification.defaultTimeStamps;
import static utilities.ArrayUtilities.mean;
import static utilities.GenericTools.linSpace;
import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.InstanceTools.truncateInstances;
import static utilities.Utilities.argMax;

public class SR1CF1 extends AbstractClassifier {

    private double alpha = 0.9;
    private int[] timeStamps;
    private int numParamValues = 100;

    private int fullLength;
    private int numInstances;

    private Classifier classifierType = new BayesNet();
    private Classifier[] classifiers;
    private double[][][] cvProbabilities;
    private double[] classValues;
    private double[] p;

    private int seed = 0;
    private Random rand;

    public SR1CF1(){}

    public void setAlpha(double d) { alpha = d; }

    public void setClassifier(Classifier c){ classifierType = c; }

    public void setSeed(int i) { seed = i; }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (timeStamps == null) timeStamps = defaultTimeStamps(data.numAttributes()-1);
        fullLength = data.numAttributes()-1;
        numInstances = data.numInstances();
        rand = new Random(seed);

        classifiers = new AbstractClassifier[timeStamps.length];
        cvProbabilities = new double[timeStamps.length][][];
        classValues = data.attributeToDoubleArray(data.classIndex());
        p = new double[3];

        for (int i = 0; i < timeStamps.length; i++){
            Instances truncatedData = truncateInstances(data, fullLength, timeStamps[i]);
            classifiers[i] = AbstractClassifier.makeCopy(classifierType);
            classifiers[i].buildClassifier(truncatedData);

            CrossValidationEvaluator cv = new CrossValidationEvaluator();
            cv.setSeed(seed);
            cv.setNumFolds(5);
            ClassifierResults r = cv.crossValidateWithStats(AbstractClassifier.makeCopy(classifierType), truncatedData);
            cvProbabilities[i] = r.getProbabilityDistributionsAsArray();
        }

        double[] pVals = linSpace(numParamValues, -1, 1);
        double bestGain = 0;
        double[] bestP = null;
        for (int i = 0; i < pVals.length; i++) {
            for (int n = 0; n < pVals.length; n++) {
                for (int g = 0; g < pVals.length; g++) {
                    p[0] = pVals[i];
                    p[1] = pVals[n];
                    p[2] = pVals[g];

                    double gain = gainFunction();
                    if (gain > bestGain || (gain == bestGain && rand.nextBoolean())){
                        bestGain = gain;
                        bestP = Arrays.copyOf(p, p.length);
                    }
                }
            }
        }

        p = bestP;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        if (probs == null) return -1;

        int maxClass = 0;
        for (int n = 1; n < probs.length; ++n) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
            else if (probs[n] == probs[maxClass]){
                if (rand.nextBoolean()){
                    maxClass = n;
                }
            }
        }

        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int idx = -1;
        for (int i = 0; i < timeStamps.length; i++){
            if (instance.numAttributes()-1 == timeStamps[i]){
                idx = i;
                break;
            }
        }
        if (idx == -1) throw new Exception("Input instance length does not match any given timestamps.");

        double[] probs = classifiers[idx].distributionForInstance(instance);

        if (idx == timeStamps.length-1 || stoppingRule(probs, timeStamps[idx])){
            return probs;
        }
        else{
            return null;
        }
    }

    private boolean stoppingRule(double[] probs, int length){
        double largestVal = -1;
        double secondLargestVal = -1;
        for (int i = 0; i < probs.length; i++){
            if (probs[i] > largestVal){
                secondLargestVal = largestVal;
                largestVal = probs[i];
            }
            else if (probs[i] > secondLargestVal){
                secondLargestVal = probs[i];
            }
        }

        return (p[0]*largestVal + p[1]*(largestVal-secondLargestVal) + p[2]*length/fullLength) > 0;
    }

    private double gainFunction() {
        double gain = 0;
        for (int i = 0; i < numInstances; i++){
            for (int n = 0; n < timeStamps.length; n++) {
                if (n == timeStamps.length-1 || stoppingRule(cvProbabilities[n][i], timeStamps[n])) {
                    gain += alpha *accuracyGain(classValues[i], cvProbabilities[n][i]) +
                            (1 - alpha)*earlinessGain(timeStamps[n]);
                    break;
                }
            }
        }
        return gain;
    }

    private double accuracyGain(double actualClass, double[] probs){
        int predClass = argMax(probs, rand);
        return actualClass == predClass ? 1 : 0;
    }

    private double earlinessGain(int length){
        return 1-length/(double)fullLength;
    }

    public static void main(String[] args) throws Exception{
        int fold = 0;
        String dataset = "Trace";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+"\\"+dataset+"_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        Random r = new Random(fold);

        SR1CF1 cls = new SR1CF1();
        cls.buildClassifier(train);
        System.out.println(Arrays.toString(cls.p));

        int length = test.numAttributes()-1;
        double[][] testProbs = new double[test.numInstances()][];
        double[] testPreds = new double[test.numInstances()];
        double[] testEarliness = new double[test.numInstances()];

        for (int i = 0; i < 20; i++){
            int newLength = (int)Math.round((i+1)*0.05 * length);
            Instances newData = truncateInstances(test, length, newLength);

            for (int n = 0; n < test.numInstances(); n++){
                if (testProbs[n] == null) {
                    Instance inst = newData.get(n);
                    double[] probs = cls.distributionForInstance(inst);

                    if (probs != null) {
                        testProbs[n] = probs;
                        testPreds[n] = argMax(probs, r);
                        testEarliness[n] = newLength/(double)length;
                    }
                }
            }
        }

        double[] trueClassVals = test.attributeToDoubleArray(test.classIndex());

        String[] stringEarliness = new String[test.numInstances()];
        for (int n = 0; n < testEarliness.length; n++){
            stringEarliness[n] = Double.toString(testEarliness[n]);
        }

        ClassifierResults cr = new ClassifierResults();
        cr.addAllPredictions(trueClassVals, testPreds, testProbs, new long[test.numInstances()], stringEarliness);

        System.out.println(mean(testEarliness));
        System.out.println(cr.getAcc());
    }

    // Code for a bayesian parameter search method.

//            BayesianSearcher bs = new BayesianSearcher(gainFunction);
//            bs.setSeed(seed);
//            ParameterSpace pspace = new ParameterSpace();
//            pspace.addParameter("p1", linSpace(50, -1, 1));
//            pspace.addParameter("p2", linSpace(50, -1, 1));
//            pspace.addParameter("p3", linSpace(50, -1, 1));
//            bs.setParameterSpace(pspace);
//            Iterator it = bs.iterator();
//
//            while(it.hasNext()){
//                it.next();
//            }
//
//            ParameterSet pset = bs.getBestParameters();
//
//            int g = 0;
//            for (Map.Entry<String, String> entry: pset.parameterSet.entrySet()) {
//                p[g] = Double.parseDouble(entry.getValue());
//                g++;
//            }


//        private Function<ParameterSet, Double> gainFunction = (ParameterSet pset) -> {
//            int g = 0;
//            for (Map.Entry<String, String> entry: pset.parameterSet.entrySet()) {
//                p[g] = Double.parseDouble(entry.getValue());
//                g++;
//            }
//
//            double gain = 0;
//            for (int i = 0; i < numInstances; i++){
//                for (int n = 0; n < timeStamps.length; n++) {
//                    if (n == timeStamps.length-1 || stoppingRule(cvProbabilities[n][i], timeStamps[n])) {
//                        gain += alpha*accuracyGain(classValues[i], cvProbabilities[n][i]) +
//                                (1 - alpha)*earlinessGain(timeStamps[n]);
//                        break;
//                    }
//                }
//            }
//            return gain;
//        };
}
