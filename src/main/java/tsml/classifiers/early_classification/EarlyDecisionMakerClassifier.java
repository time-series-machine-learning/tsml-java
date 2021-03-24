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

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.interval_based.TSF;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import java.util.Arrays;
import java.util.Random;

import static utilities.ArrayUtilities.mean;
import static utilities.InstanceTools.*;
import static utilities.Utilities.argMax;

/**
 * Early classification classifier which makes use of a tsml/weka classifier and decision maker.
 * Can load from test and train files.
 *
 * @author Matthew Middlehurst
 */
public class EarlyDecisionMakerClassifier extends AbstractEarlyClassifier implements Randomizable {

    private Classifier classifier;
    private EarlyDecisionMaker decisionMaker;

    private Classifier[] classifiers;

    private int seed = 0;
    private Random rand;

    private boolean loadFromFile = false;
    private String loadPath;
    private ClassifierResults[] loadedResults;
    private int testInstanceCounter = 0;
    private int lastIdx = -1;

    public EarlyDecisionMakerClassifier(Classifier classifier, EarlyDecisionMaker decisionMaker){
        this.classifier = classifier;
        this.decisionMaker = decisionMaker;
    }

    @Override
    public int getSeed() { return 0; }

    @Override
    public void setSeed(int i) { seed = i; }

    public void setLoadFromFilePath(String path) {
        loadFromFile = true;
        loadPath = path;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (thresholds == null) thresholds = decisionMaker.defaultTimeStamps(data.numAttributes()-1);
        classifiers = new Classifier[thresholds.length];
        if (classifier instanceof Randomizable) ((Randomizable) classifier).setSeed(seed);
        if (decisionMaker instanceof Randomizable) ((Randomizable) decisionMaker).setSeed(seed);
        decisionMaker.setNormalise(normalise);
        rand = new Random(seed);

        if (loadFromFile) {
            loadedResults = new ClassifierResults[thresholds.length];
            for (int i = 0; i < thresholds.length; i++) {
                loadedResults[i] = new ClassifierResults(loadPath + thresholds[i] + "testFold" + seed +
                        ".csv");
            }

            if (decisionMaker instanceof LoadableEarlyDecisionMaker){
                ((LoadableEarlyDecisionMaker) decisionMaker).loadFromFile(data, loadPath, thresholds);
            }
            else{
                decisionMaker.fit(data, classifier, thresholds);
            }
        }
        else {
            int length = data.numAttributes() - 1;
            for (int i = 0; i < thresholds.length; i++) {
                if (thresholds[i] < 3) {
                    thresholds = Arrays.copyOfRange(thresholds, 1, thresholds.length);
                    classifiers = new Classifier[thresholds.length];
                    i--;
                    continue;
                }

                Instances newData = truncateInstances(data, length, thresholds[i]);
                if (normalise) newData = zNormaliseWithClass(newData);

                classifiers[i] = AbstractClassifier.makeCopy(classifier);
                classifiers[i].buildClassifier(newData);
            }

            Classifier[] blankClassifiers = new Classifier[thresholds.length];
            for (int i = 0; i < blankClassifiers.length; i++) {
                blankClassifiers[i] = AbstractClassifier.makeCopy(classifier);
            }
            decisionMaker.fit(data, blankClassifiers, thresholds);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        return probs == null ? -1 : argMax(probs, rand);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int idx = -1;
        for (int i = 0; i < thresholds.length; i++){
            if (instance.numAttributes()-1 == thresholds[i]){
                idx = i;
                break;
            }
        }
        if (idx == -1) throw new Exception("Input instance length does not match any given timestamps.");

        double[] probs;
        boolean decision;

        Instance newData = instance;
        if (normalise) newData = zNormaliseWithClass(instance);

        if (loadFromFile){
            probs = loadedResults[idx].getProbabilityDistribution(testInstanceCounter);
            decision = decisionMaker.decide(idx, probs);
            if (idx <= lastIdx) testInstanceCounter++;
            lastIdx = idx;
        }
        else {
            probs = classifiers[idx].distributionForInstance(newData);
            decision = decisionMaker.decide(idx, probs);
        }

        return decision ? probs : null;
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;
        String dataset = "ItalyPowerDemand";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "\\"
                + dataset + "_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "\\"
                + dataset + "_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        Random r = new Random(fold);

        Classifier c = new TSF();
        EarlyDecisionMaker dm = new TEASER();

        EarlyDecisionMakerClassifier cls = new EarlyDecisionMakerClassifier(c, dm);
        cls.normalise = true;
        cls.buildClassifier(train);

        int length = test.numAttributes() - 1;
        double[][] testProbs = new double[test.numInstances()][];
        double[] testPreds = new double[test.numInstances()];
        double[] testEarliness = new double[test.numInstances()];

        for (int i = 0; i < cls.thresholds.length; i++) {
            Instances newData = truncateInstances(test, length, cls.thresholds[i]);

            for (int n = 0; n < test.numInstances(); n++) {
                if (testProbs[n] == null) {
                    Instance inst = newData.get(n);
                    double[] probs = cls.distributionForInstance(inst);

                    if (probs != null) {
                        testProbs[n] = probs;
                        testPreds[n] = argMax(probs, r);
                        testEarliness[n] =  cls.thresholds[i] / (double) length;
                    }
                }
            }
        }

        double[] trueClassVals = test.attributeToDoubleArray(test.classIndex());

        String[] stringEarliness = new String[test.numInstances()];
        for (int n = 0; n < testEarliness.length; n++) {
            stringEarliness[n] = Double.toString(testEarliness[n]);
        }

        ClassifierResults cr = new ClassifierResults();
        cr.addAllPredictions(trueClassVals, testPreds, testProbs, new long[test.numInstances()], stringEarliness);
        double accuracy = cr.getAcc();
        double earliness = mean(testEarliness);

        System.out.println("Accuracy  " + accuracy);
        System.out.println("Earliness " + earliness);
        System.out.println("HM        " + (2 * accuracy * (1 - earliness)) / (accuracy + (1 - earliness)));
    }
}
