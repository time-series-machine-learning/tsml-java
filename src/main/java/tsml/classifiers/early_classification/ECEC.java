package tsml.classifiers.early_classification;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.EnhancedAbstractClassifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Randomizable;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static utilities.ArrayUtilities.unique;
import static utilities.InstanceTools.truncateInstances;
import static utilities.InstanceTools.zNormaliseWithClass;
import static utilities.Utilities.argMax;

public class ECEC extends EarlyDecisionMaker implements Randomizable, LoadableEarlyDecisionMaker {

    private double ratio = 0.8;

    private double confidenceThreshold;
    private int finalIndex;

    private int[] labels;
    private int[][] predCount;
    private int[][] correctCount;

    private int seed;
    private Random rand;

    public ECEC() {
    }

    @Override
    public void setSeed(int s) {
        seed = s;
    }

    @Override
    public int getSeed() {
        return seed;
    }

    @Override
    public void fit(Instances data, Classifier[] classifiers, int[] thresholds) throws Exception {
        double[][][] trainProbabilities = new double[thresholds.length][][];

        for (int i = 0; i < thresholds.length; i++) {
            if (classifiers[i] instanceof EnhancedAbstractClassifier &&
                    ((EnhancedAbstractClassifier) classifiers[i]).ableToEstimateOwnPerformance() &&
                    ((EnhancedAbstractClassifier) classifiers[i]).getEstimateOwnPerformance()) {
                trainProbabilities[i] = ((EnhancedAbstractClassifier) classifiers[i]).getTrainResults()
                        .getProbabilityDistributionsAsArray();
            } else {
                Instances truncatedData = truncateInstances(data, data.numAttributes() - 1, thresholds[i]);
                if (normalise) zNormaliseWithClass(truncatedData);

                CrossValidationEvaluator cv = new CrossValidationEvaluator();
                cv.setSeed(seed);
                cv.setNumFolds(5);
                ClassifierResults r = cv.crossValidateWithStats(AbstractClassifier.makeCopy(classifiers[i]),
                        truncatedData);
                trainProbabilities[i] = r.getProbabilityDistributionsAsArray();
            }
        }

        fitECEC(data, thresholds, trainProbabilities);
    }

    @Override
    public void loadFromFile(Instances data, String directoryPath, int[] thresholds) throws Exception {
        double[][][] trainProbabilities = new double[thresholds.length][][];

        for (int i = 0; i < thresholds.length; i++) {
            ClassifierResults r = new ClassifierResults(directoryPath + thresholds[i] + "trainFold" +
                    seed + ".csv");
            trainProbabilities[i] = r.getProbabilityDistributionsAsArray();
        }

        fitECEC(data, thresholds, trainProbabilities);
    }

    @Override
    public boolean decide(int thresholdIndex, double[] probabilities) throws Exception {
        if (thresholdIndex == finalIndex) return true;
        if (thresholdIndex == 0) labels = new int[finalIndex + 1];

        labels[thresholdIndex] = argMax(probabilities, rand);
        double mod = 1;
        for(int j = 0; j <= thresholdIndex; j++) {
            if (labels[j] == labels[thresholdIndex]) {
                double correct = (double) correctCount[j][labels[thresholdIndex]] /
                        predCount[j][labels[thresholdIndex]];
                mod *= 1 - correct;
            }
        }
        double confidence = 1 - mod;

        return confidence >= confidenceThreshold;
    }

    public void fitECEC(Instances data, int[] thresholds, double[][][] trainProbabilities) throws Exception {
        rand = new Random(seed);
        finalIndex = thresholds.length - 1;
        int seriesLength = data.numAttributes() - 1;

        int[][] trainPred = new int[thresholds.length][data.numInstances()];
        predCount = new int[thresholds.length][data.numClasses()];
        correctCount = new int[thresholds.length][data.numClasses()];

        for (int n = 0; n < thresholds.length; n++) {
            for (int i = 0; i < data.numInstances(); i++) {
                trainPred[n][i] = argMax(trainProbabilities[n][i], rand);
                predCount[n][trainPred[n][i]]++;

                if (trainPred[n][i] == data.get(i).classIndex()) {
                    correctCount[n][trainPred[n][i]]++;
                }
            }
        }

        double[][] confidences = new double[thresholds.length][data.numInstances()];
        double[] allConfidences = new double[thresholds.length * data.numInstances()];
        ArrayList<Double>[] classCondifences = new ArrayList[data.numClasses()];
        for (int i = 0; i < classCondifences.length; i++) {
            classCondifences[i] = new ArrayList<>();
        }

        int p = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            for (int n = 0; n < thresholds.length; n++) {
                double mod = 1;
                for (int j = 0; j <= n; j++) {
                    if (trainPred[j][i] == trainPred[n][i]) {
                        double correct = (double) correctCount[j][trainPred[n][i]] / predCount[j][trainPred[n][i]];
                        mod *= 1 - correct;
                    }
                }
                confidences[n][i] = 1 - mod;

                allConfidences[p++] = confidences[n][i];
                classCondifences[(int) data.get(i).classValue()].add(confidences[n][i]);
            }
        }

        List<Double> uniqueConfidences = unique(allConfidences);

        double[] confidenceThresholds = new double[uniqueConfidences.size() - 1];
        for (int i = 0; i < confidenceThresholds.length; i++) {
            confidenceThresholds[i] = (uniqueConfidences.get(i) + uniqueConfidences.get(i + 1)) / 2;
        }

        double minCost = Double.MAX_VALUE;
        for (double threshold : confidenceThresholds) {
            int success = 0;
            double earliness = 0;
            for (int n = 0; n < data.numInstances(); n++) {
                for (int j = 0; j < thresholds.length; j++) {
                    if (confidences[n][j] > threshold || j == finalIndex) {
                        earliness += (double) thresholds[j] / seriesLength;
                        if (trainPred[n][j] == (int) data.get(n).classValue()) {
                            success++;
                        }
                        break;
                    }
                }
            }

            double cost = ratio * (data.numInstances() - success) + (1 - ratio) * earliness;

            if (cost < minCost) {
                minCost = cost;
                confidenceThreshold = threshold;
            }
        }
    }
}
