package tsml.classifiers.early_classification;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.interval_based.TSF;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static experiments.ExperimentsEarlyClassification.defaultTimeStamps;
import static utilities.InstanceTools.zNormaliseWithClass;
import static utilities.Utilities.argMax;

public class ProbabilityThreshold extends AbstractClassifier {

    private Classifier classifier = new TSF();
    private double threshold = 0.9;
    private boolean addTruncatedTrainInstances = false;
    private int[] truncationTimeStamps;
    private int extraTrainInstances = -1;

    private Instances dataHeader;
    private int seriesLength;

    private int seed = 0;
    private Random rand;

    public ProbabilityThreshold() { }

    public void setClassifier(Classifier c) {
        classifier = c;
    }

    public void setThreshold(double d) { threshold = d; }

    public void setSeed(int i){ seed = i; }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        data = zNormaliseWithClass(data);

        if (addTruncatedTrainInstances){
            if (truncationTimeStamps == null) truncationTimeStamps = defaultTimeStamps(data.numAttributes()-1);
            if (extraTrainInstances == -1) extraTrainInstances = truncationTimeStamps.length * data.numInstances();

            //todo
            //shortenData with normalisation
        }

        classifier.buildClassifier(data);
        rand = new Random(seed);
        dataHeader = new Instances(data, 0);
        seriesLength = data.numAttributes()-1;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        if (probs == null) return -1;

        int maxClass = 0;
        for (int n = 1; n < probs.length; ++n) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            } else if (probs[n] == probs[maxClass]) {
                if (rand.nextBoolean()) {
                    maxClass = n;
                }
            }
        }

        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        boolean fullSeries = true;
        int instLength = instance.numAttributes()-1;
        if (instLength < seriesLength){
            double[] series = Arrays.copyOf(instance.toDoubleArray(), seriesLength+1);
            series[seriesLength] = series[instLength];
            series[instLength] = 0;
            instance = new DenseInstance(1, series);
            instance.setDataset(dataHeader);
            fullSeries = false;
        }

        instance = zNormaliseWithClass(instance);
        double[] probs = classifier.distributionForInstance(instance);

        if (fullSeries || argMax(probs, rand) > threshold) {
            return probs;
        } else {
            return null;
        }
    }

    public static ClassifierResults resultsFromFile(String resultsPath, int foldNo, double threshold, Instances data)
            throws Exception {
        ClassifierResults res = new ClassifierResults(resultsPath + "/5%testFold" + foldNo + ".csv");

        ClassifierResults newRes = new ClassifierResults(res.numClasses());
        newRes.setTimeUnit(TimeUnit.NANOSECONDS);
        newRes.setClassifierName(res.getClassifierName());
        newRes.setDatasetName(res.getDatasetName());
        newRes.setFoldID(foldNo);
        newRes.setSplit("test");

        int length = data.numAttributes()-1;
        newRes.turnOffZeroTimingsErrors();

        double[] trueClassVals = data.attributeToDoubleArray(data.classIndex());
        double[] predictions = new double[data.numInstances()];
        double[][] distributions = new double[data.numInstances()][];
        long[] predTimes = new long[data.numInstances()];
        String[] descriptions = new String[data.numInstances()];

        Random rand = new Random(foldNo);

        for (int n = 5; n <= 100; n += 5) {
            res = new ClassifierResults(resultsPath + "/" + n + "%testFold" + foldNo + ".csv");

            for (int i = 0; i < data.numInstances(); i++) {
                if (distributions[i] == null) {
                    double[] dist = res.getProbabilityDistribution(i);
                    predTimes[i] += res.getPredictionTime(i);
                    int maxIndex = argMax(dist, rand);

                    if (dist[maxIndex] > threshold) {
                        int newLength = (int) Math.round((n / 100.0 * length));
                        descriptions[i] = Double.toString(newLength / (double) length);
                        distributions[i] = dist;
                        predictions[i] = maxIndex;
                    }
                }
            }
        }

        newRes.addAllPredictions(trueClassVals, predictions, distributions, predTimes, descriptions);
        newRes.turnOnZeroTimingsErrors();

        newRes.finaliseResults();
        newRes.findAllStatsOnce();

        return newRes;
    }
}
