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
import tsml.transformers.ShapeletTransform;
import tsml.transformers.shapelet_tools.ShapeletTransformFactory;
import tsml.transformers.shapelet_tools.ShapeletTransformFactoryOptions;
import tsml.transformers.shapelet_tools.ShapeletTransformTimingUtilities;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

import static utilities.ArrayUtilities.mean;
import static utilities.InstanceTools.*;
import static utilities.Utilities.argMax;

/**
 * Early classification classifier using the shapelet transform.
 * Extracts shapelets from the full series and uses them and a decision maker to classify.
 *
 * @author Matthew Middlehurst
 */
public class ShapeletTransformEarlyClassifier extends AbstractEarlyClassifier {

    private Classifier classifier;
    private EarlyDecisionMaker decisionMaker = new ProbabilityThreshold();

    private ShapeletTransform transform;
    private Instances shapeletData;
    private int[] redundantFeatures;

    private int seed;
    private Random rand;

    public ShapeletTransformEarlyClassifier() { }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        rand = new Random(seed);
        decisionMaker.setNormalise(normalise);
        int n = data.numInstances();
        int m = data.numAttributes()-1;

        Instances newData = data;
        if (normalise) newData = zNormaliseWithClass(data);

        ShapeletSearch.SearchType searchType = ShapeletSearch.SearchType.RANDOM;
        ShapeletTransformFactoryOptions.ShapeletTransformOptions transformOptions
                = new ShapeletTransformFactoryOptions.ShapeletTransformOptions();
        transformOptions.setDistanceType(ShapeletDistance.DistanceType.NORMAL); //default STC uses improved online
        transformOptions.setQualityMeasure(ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN);
        transformOptions.setRescalerType(ShapeletDistance.RescalerType.NORMALISATION);
        transformOptions.setRoundRobin(true);
        transformOptions.setCandidatePruning(true);
        transformOptions.setMinLength(3);
        transformOptions.setMaxLength(data.numAttributes()-1);
        if(data.numClasses() > 2) {
            transformOptions.setBinaryClassValue(true);
            transformOptions.setClassBalancing(true);
        }else{
            transformOptions.setBinaryClassValue(false);
            transformOptions.setClassBalancing(false);
        }
        int numShapeletsInTransform= Math.min(10 * n, ShapeletTransform.MAXTRANSFORMSIZE);
        transformOptions.setKShapelets(numShapeletsInTransform);

        long numShapeletsInProblem = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(n, m,
                3, m);
        long numShapeletsToEvaluate = 100000; //hardcoded for now
        if (numShapeletsToEvaluate < n) {
            numShapeletsToEvaluate = n;
        }
        if (numShapeletsToEvaluate >= numShapeletsInProblem){
            numShapeletsToEvaluate = numShapeletsInProblem;
            searchType = ShapeletSearch.SearchType.FULL;
        }

        ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
        searchBuilder.setSeed(2*seed);
        searchBuilder.setMin(transformOptions.getMinLength());
        searchBuilder.setMax(transformOptions.getMaxLength());
        searchBuilder.setSearchType(searchType);
        searchBuilder.setNumShapeletsToEvaluate(numShapeletsToEvaluate/n);
        transformOptions.setSearchOptions(searchBuilder.build());

        transform = new ShapeletTransformFactory(transformOptions.build()).getTransform();
        shapeletData = transform.fitTransform(newData);
        redundantFeatures = InstanceTools.removeRedundantTrainAttributes(shapeletData);

        RotationForest rotf = new RotationForest();
        rotf.setNumIterations(200);
        rotf.setSeed(seed);
        classifier = rotf;
        classifier.buildClassifier(shapeletData);

        thresholds = decisionMaker.defaultTimeStamps(data.numAttributes()-1);
        decisionMaker.fit(newData, classifier, thresholds);

        shapeletData = new Instances(data,0);
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

        Instance newData = instance;
        if (normalise) newData = zNormaliseWithClass(instance);

        shapeletData = new Instances(instance.dataset(),0);
        shapeletData.add(newData);
        Instances temp = transform.transform(shapeletData);

        for (int del: redundantFeatures)
            temp.deleteAttributeAt(del);

        double[] probs = classifier.distributionForInstance(temp.get(0));
        boolean decision = decisionMaker.decide(idx, probs);

        return decision ? probs : null;
    }

    public static void main(String[] args) throws Exception{
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

        ShapeletTransformEarlyClassifier cls = new ShapeletTransformEarlyClassifier();
        cls.seed = fold;
        cls.normalise = true;
        cls.buildClassifier(train);

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
}
