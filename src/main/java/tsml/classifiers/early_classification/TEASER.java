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

import com.carrotsearch.hppc.IntIntHashMap;
import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.EnhancedAbstractClassifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import static utilities.InstanceTools.truncateInstances;
import static utilities.InstanceTools.zNormaliseWithClass;
import static utilities.Utilities.argMax;

/**
 * TEASER early classification decision maker.
 * Trains a 1-class SVM for each threshold, requires v positive decisions in a row to return a true decision.
 *
 * Schäfer, Patrick, and Ulf Leser. "Teaser: Early and accurate time series classification."
 * Data Mining and Knowledge Discovery (2020): 1-27.
 * https://link.springer.com/article/10.1007/s10618-020-00690-z
 *
 * @author Matthew Middlehurst
 */
public class TEASER extends EarlyDecisionMaker implements Randomizable, LoadableEarlyDecisionMaker {

    private static final double[] SVM_GAMMAS = new double[]{100, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1};

    private LibSVM[] svm;
    private int[] timeStamps;
    private Instances probDataHeader;
    private IntIntHashMap predCounts;
    private int v;

    private int seed;
    private Random rand;

    public TEASER() {}

    @Override
    public void setSeed(int s) { seed = s; }

    @Override
    public int getSeed() { return seed; }

    @Override
    public void fit(Instances data, Classifier[] classifiers, int[] thresholds) throws Exception {
        double[][][] trainProbabilities = new double[thresholds.length][][];

        for (int i = 0; i < thresholds.length; i++) {
            if (classifiers[i] instanceof EnhancedAbstractClassifier &&
                    ((EnhancedAbstractClassifier) classifiers[i]).ableToEstimateOwnPerformance() &&
                    ((EnhancedAbstractClassifier) classifiers[i]).getEstimateOwnPerformance()) {
                trainProbabilities[i] = ((EnhancedAbstractClassifier) classifiers[i]).getTrainResults()
                        .getProbabilityDistributionsAsArray();
            }
            else {
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

        fitTEASER(data, thresholds, trainProbabilities);
    }

    @Override
    public void loadFromFile(Instances data, String directoryPath, int[] thresholds) throws Exception {
        double[][][] trainProbabilities = new double[thresholds.length][][];

        for (int i = 0; i < thresholds.length; i++) {
            ClassifierResults r = new ClassifierResults(directoryPath + thresholds[i] + "trainFold" +
                    seed + ".csv");
            trainProbabilities[i] = r.getProbabilityDistributionsAsArray();
        }

        fitTEASER(data, thresholds, trainProbabilities);
    }

    @Override
    public boolean decide(int thresholdIndex, double[] probabilities) throws Exception {
        if (thresholdIndex == timeStamps.length-1) return true;

        int pred = argMax(probabilities, rand);
        double minDiff = 1;
        for (int g = 0; g < probabilities.length; g++) {
            if (pred != g) {
                minDiff = Math.min(minDiff, probabilities[pred] - probabilities[g]);
            }
        }

        double[] arr = Arrays.copyOf(probabilities, probabilities.length + 2);
        arr[arr.length - 2] = minDiff;
        Instance inst = new DenseInstance(1, arr);
        inst.setDataset(probDataHeader);

        if (thresholdIndex == 0) predCounts = new IntIntHashMap();

        if (svm[thresholdIndex].distributionForInstance(inst)[0] == 1) {
            int count = predCounts.get(pred);
            if (count == 0) {
                predCounts.clear();
                predCounts.put(pred, 1);
            } else {
                count++;
                if (count >= v) {
                    predCounts.clear();
                    return true;
                } else {
                    predCounts.put(pred, count);
                    return false;
                }
            }
        }

        return false;
    }

    public void fitTEASER(Instances data, int[] thresholds, double[][][] trainProbabilities)
            throws Exception {
        rand = new Random(seed);
        libsvm.svm.rand.setSeed(seed);
        // Disables svm output
        libsvm.svm.svm_set_print_string_function(s -> { });
        timeStamps = thresholds;
        svm = new LibSVM[thresholds.length];

        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 1; i <= data.numClasses()+1; i++) {
            atts.add(new Attribute("att" + i));
        }
        ArrayList<String> cls = new ArrayList<>(1);
        cls.add("1");
        atts.add(new Attribute("cls", cls));
        probDataHeader = new Instances("probData", atts, 0);
        probDataHeader.setClassIndex(probDataHeader.numAttributes()-1);

        Instances[] trainData = new Instances[thresholds.length];
        int[][] trainPred = new int[thresholds.length][data.numInstances()];

        for (int i = 0; i < thresholds.length; i++) {
            trainData[i] = new Instances(probDataHeader, data.numInstances());

            Instances probData = new Instances(probDataHeader, data.numInstances());
            for (int n = 0; n < data.numInstances(); n++){
                trainPred[i][n] = argMax(trainProbabilities[i][n], rand);
                double minDiff = 1;
                for (int g = 0; g < trainProbabilities[i][n].length; g++) {
                    if (trainPred[i][n] != g) {
                        minDiff = Math.min(minDiff, trainProbabilities[i][n][trainPred[i][n]] -
                                trainProbabilities[i][n][g]);
                    }
                }

                double[] arr = Arrays.copyOf(trainProbabilities[i][n], trainProbabilities[i][n].length + 2);
                arr[arr.length-2] = minDiff;
                Instance inst = new DenseInstance(1, arr);

                trainData[i].add(inst);
                if (trainPred[i][n] == data.get(n).classValue()) {
                    probData.add(inst);
                }
            }

            probData.randomize(rand);

            double bestAccuracy = -1;
            for (double svmGamma : SVM_GAMMAS) {
                LibSVM svmCandidate = new LibSVM();
                svmCandidate.setSVMType(new SelectedTag(LibSVM.SVMTYPE_ONE_CLASS_SVM, LibSVM.TAGS_SVMTYPE));
                svmCandidate.setEps(1e-4);
                svmCandidate.setGamma(svmGamma);
                svmCandidate.setNu(0.05);
                svmCandidate.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
                svmCandidate.setCacheSize(40);

                double correct = 0;
                for (int n = 0; n < 10; n++){
                    Instances cvTrain = probData.trainCV(10, n);
                    Instances cvTest = probData.testCV(10, n);
                    LibSVM svmCV = (LibSVM)AbstractClassifier.makeCopy(svmCandidate);
                    svmCV.buildClassifier(cvTrain);

                    for (Instance inst: cvTest){
                        if (svmCV.distributionForInstance(inst)[0] == 1){
                            correct++;
                        }
                    }
                }
                double accuracy = correct / probData.numInstances();

                if (accuracy > bestAccuracy){
                    svm[i] = svmCandidate;
                    bestAccuracy = accuracy;
                }
            }

            svm[i].buildClassifier(probData);
        }

        double bestHM = -1;
        for (int g = 2; g <= 5; g++) {
            double correctSum = 0;
            double earlinessSum = 0;
            for (int n = 0; n < data.numInstances(); n++){
                IntIntHashMap counts = new IntIntHashMap();
                for (int i = 0; i < thresholds.length; i++){
                    if (svm[i].distributionForInstance(trainData[i].get(n))[0] == 1 || i == thresholds.length-1){
                        int count = counts.get(trainPred[i][n]);
                        if (count == 0 && i < thresholds.length-1){
                            counts.clear();
                            counts.put(trainPred[i][n], 1);
                        }
                        else{
                            count++;
                            if (count >= g || i == thresholds.length-1){
                                if (trainPred[i][n] == data.get(n).classValue())
                                    correctSum++;
                                earlinessSum += thresholds[i] / (data.numAttributes()-1.0);
                                break;
                            }
                            else {
                                counts.put(trainPred[i][n], count);
                            }
                        }
                    }
                }
            }

            double accuracy = correctSum / data.numInstances();
            double earliness = 1.0 - earlinessSum / data.numInstances();
            double hm = (2 * accuracy * earliness) / (accuracy + earliness);

            if (hm > bestHM) {
                bestHM = hm;
                v = g;
            }
        }
    }
}
