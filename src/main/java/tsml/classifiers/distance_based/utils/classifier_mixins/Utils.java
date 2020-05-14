package tsml.classifiers.distance_based.utils.classifier_mixins;

import evaluation.storage.ClassifierResults;
import java.util.Arrays;
import java.util.logging.Level;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.logging.Loggable;
import utilities.ArrayUtilities;
import utilities.InstanceTools;
import utilities.Utilities;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class Utils {


    public static void trainTestPrint(Classifier classifier, Instances[] trainAndTestData) throws Exception {
        if(classifier instanceof Loggable) {
            ((Loggable) classifier).getLogger().setLevel(Level.ALL);
        }
        final Instances trainData = trainAndTestData[0];
        final Instances testData = trainAndTestData[1];
        classifier.buildClassifier(trainData);
        if(classifier instanceof EnhancedAbstractClassifier) {
            if(((EnhancedAbstractClassifier) classifier).getEstimateOwnPerformance()) {
                ClassifierResults trainResults = ((EnhancedAbstractClassifier) classifier).getTrainResults();
                trainResults.setDetails(classifier, trainData);
                System.out.println("train results:");
                System.out.println(trainResults.writeSummaryResultsToString());
            }
        }
        ClassifierResults testResults = new ClassifierResults();
        for(Instance instance : testData) {
            addPrediction(classifier, instance, testResults);
        }
        testResults.setDetails(classifier, trainData);
        System.out.println("test results:");
        System.out.println(testResults.writeSummaryResultsToString());
    }

    public static void addPrediction(Classifier classifier, Instance test, ClassifierResults results) throws Exception {
        final double classValue = test.classValue();
        test.setClassMissing();
        long timestamp = System.nanoTime();
        final double[] distribution = classifier.distributionForInstance(test);
        long testTime = System.nanoTime() - timestamp;
        if(classifier instanceof TestTimeable) {
            testTime = ((TestTimeable) classifier).getTestTimeNanos();
        }
        final double prediction = ArrayUtilities.argMax(distribution);
        results.addPrediction(classValue, distribution, prediction, testTime, null);
        test.setClassValue(classValue);
    }
}
