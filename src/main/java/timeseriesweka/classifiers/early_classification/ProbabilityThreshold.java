package timeseriesweka.classifiers.early_classification;

import timeseriesweka.classifiers.interval_based.TSF;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;

import static experiments.ExperimentsEarlyClassification.defaultTimeStamps;
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

    public void setThreshold(double d) {
        threshold = d;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (addTruncatedTrainInstances){
            if (truncationTimeStamps == null) truncationTimeStamps = defaultTimeStamps(data.numAttributes()-1);
            if (extraTrainInstances == -1) extraTrainInstances = truncationTimeStamps.length * data.numInstances();

            //todo
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

        double[] probs = classifier.distributionForInstance(instance);

        if (fullSeries || argMax(probs, rand) > threshold) {
            return probs;
        } else {
            return null;
        }
    }
}
