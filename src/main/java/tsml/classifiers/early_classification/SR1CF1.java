/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers.early_classification;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.EnhancedAbstractClassifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Randomizable;

import java.util.Arrays;
import java.util.Random;

import static utilities.GenericTools.linSpace;
import static utilities.InstanceTools.*;
import static utilities.Utilities.argMax;

/**
 * Modified SR1CF1 decision maker. Uses gridsearch for parameter values instead of the genetic algorithm used in the
 * paper.
 * Tunes parameters to find a suitable accuracy and earliness balance, one can be favoured over the other using the
 * alpha parameter.
 *
 * Mori, Usue, et al. "Early classification of time series by simultaneously optimizing the accuracy and earliness."
 * IEEE transactions on neural networks and learning systems 29.10 (2017): 4569-4578.
 *
 * @author Matthew Middlehurst
 */
public class SR1CF1 extends EarlyDecisionMaker implements Randomizable, LoadableEarlyDecisionMaker {

    private double alpha = 0.8;
    private int[] timeStamps;
    private int numParamValues = 200;

    private int fullLength;
    private int numInstances;

    private double[][][] cvProbabilities;
    private double[] classValues;
    private double[] p;

    private int seed = 0;
    private Random rand;

    public SR1CF1() {}

    @Override
    public int getSeed() { return seed; }

    public void setAlpha(double d) { alpha = d; }

    @Override
    public void setSeed(int i) { seed = i; }

    @Override
    public void fit(Instances data, Classifier[] classifiers, int[] thresholds) throws Exception {
        fullLength = data.numAttributes()-1;
        numInstances = data.numInstances();
        timeStamps = thresholds;
        rand = new Random(seed);

        cvProbabilities = new double[timeStamps.length][][];
        classValues = data.attributeToDoubleArray(data.classIndex());
        p = new double[3];

        for (int i = 0; i < timeStamps.length; i++) {
            if (classifiers[i] instanceof EnhancedAbstractClassifier &&
                    ((EnhancedAbstractClassifier) classifiers[i]).ableToEstimateOwnPerformance() &&
                    ((EnhancedAbstractClassifier) classifiers[i]).getEstimateOwnPerformance()) {
                cvProbabilities[i] = ((EnhancedAbstractClassifier) classifiers[i]).getTrainResults()
                        .getProbabilityDistributionsAsArray();
            }
            else {
                Instances truncatedData = truncateInstances(data, fullLength, timeStamps[i]);
                if (normalise) zNormaliseWithClass(truncatedData);

                CrossValidationEvaluator cv = new CrossValidationEvaluator();
                cv.setSeed(seed);
                cv.setNumFolds(5);
                ClassifierResults r = cv.crossValidateWithStats(AbstractClassifier.makeCopy(classifiers[i]),
                        truncatedData);
                cvProbabilities[i] = r.getProbabilityDistributionsAsArray();
            }
        }

        findP();
    }

    @Override
    public void loadFromFile(Instances data, String directoryPath, int[] thresholds) throws Exception {
        fullLength = data.numAttributes()-1;
        numInstances = data.numInstances();
        timeStamps = thresholds;
        rand = new Random(seed);

        cvProbabilities = new double[timeStamps.length][][];
        classValues = data.attributeToDoubleArray(data.classIndex());
        p = new double[3];

        for (int i = 0; i < timeStamps.length; i++) {
            ClassifierResults r = new ClassifierResults(directoryPath + thresholds[i] + "trainFold" +
                    seed + ".csv");
            cvProbabilities[i] = r.getProbabilityDistributionsAsArray();
        }

        findP();
    }

    @Override
    public boolean decide(int thresholdIndex, double[] probabilities) {
        return thresholdIndex == timeStamps.length - 1 || stoppingRule(probabilities, timeStamps[thresholdIndex]);
    }

    private void findP(){
        double[] pVals = linSpace(numParamValues, -1, 1);
        double bestGain = 0;
        double[] bestP = null;
        for (double v : pVals) {
            for (double b : pVals) {
                for (double n : pVals) {
                    p[0] = v;
                    p[1] = b;
                    p[2] = n;

                    double gain = gainFunction();
                    if (gain > bestGain || (gain == bestGain && rand.nextBoolean())) {
                        bestGain = gain;
                        bestP = Arrays.copyOf(p, p.length);
                    }
                }
            }
        }

        p = bestP;
    }

    private boolean stoppingRule(double[] probs, int length) {
        double largestVal = -1;
        double secondLargestVal = -1;
        for (double prob : probs) {
            if (prob > largestVal) {
                secondLargestVal = largestVal;
                largestVal = prob;
            } else if (prob > secondLargestVal) {
                secondLargestVal = prob;
            }
        }

        return (p[0] * largestVal + p[1] * (largestVal - secondLargestVal) + p[2] * length / fullLength) > 0;
    }

    private double gainFunction() {
        double gain = 0;
        for (int i = 0; i < numInstances; i++) {
            for (int n = 0; n < timeStamps.length; n++) {
                if (n == timeStamps.length - 1 || stoppingRule(cvProbabilities[n][i], timeStamps[n])) {
                    gain += alpha * accuracyGain(classValues[i], cvProbabilities[n][i]) +
                            (1 - alpha) * earlinessGain(timeStamps[n]);
                    break;
                }
            }
        }
        return gain;
    }

    private double accuracyGain(double actualClass, double[] probs) {
        int predClass = argMax(probs, rand);
        return actualClass == predClass ? 1 : 0;
    }

    private double earlinessGain(int length) {
        return 1 - length / (double) fullLength;
    }
}
