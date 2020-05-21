package tsml.classifiers.distance_based.utils.classifier_mixins;

import com.google.common.testing.GcFinalization;
import evaluation.storage.ClassifierResults;
import java.util.Date;
import java.util.logging.Level;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.logging.Loggable;
import tsml.classifiers.distance_based.utils.results.ResultUtils;
import tsml.classifiers.distance_based.utils.stopwatch.StopWatch;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import utilities.ArrayUtilities;
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
        MemoryWatcher overallMemoryWatcher = new MemoryWatcher();
        StopWatch overallTimer = new StopWatch();
        overallMemoryWatcher.resetAndEnable();
        overallTimer.resetAndEnable();
        MemoryWatcher memoryWatcher = new MemoryWatcher();
        StopWatch timer = new StopWatch();
        if(classifier instanceof Loggable) {
            ((Loggable) classifier).getLogger().setLevel(Level.ALL);
        }
        final Instances trainData = trainAndTestData[0];
        final Instances testData = trainAndTestData[1];
        timer.resetAndEnable();
        memoryWatcher.resetAndEnable();
        classifier.buildClassifier(trainData);
        timer.disable();
        memoryWatcher.disable();
        System.out.println("end build");
        System.out.println("train time: " + timer.getTimeNanos());
        System.out.println("train mem: " + memoryWatcher.toString());
//        GcFinalization.awaitFullGc();
        if(classifier instanceof EnhancedAbstractClassifier) {
            if(((EnhancedAbstractClassifier) classifier).getEstimateOwnPerformance()) {
                ClassifierResults trainResults = ((EnhancedAbstractClassifier) classifier).getTrainResults();
                ResultUtils.setInfo(trainResults, classifier, trainData);
                System.out.println("train results:");
                System.out.println(trainResults.writeSummaryResultsToString());
            }
        }
        timer.resetAndEnable();
        memoryWatcher.resetAndEnable();
        ClassifierResults testResults = new ClassifierResults();
        for(Instance instance : testData) {
            addPrediction(classifier, instance, testResults);
        }
        memoryWatcher.disable();
        timer.disable();
        ResultUtils.setInfo(testResults, classifier, trainData);
        System.out.println("test time: " + timer.getTimeNanos());
        System.out.println("test mem: " + memoryWatcher.toString());
        System.out.println("test results:");
        System.out.println(testResults.writeSummaryResultsToString());
        overallMemoryWatcher.disable();
        overallTimer.disable();
        System.out.println("overall time: " + overallTimer.getTimeNanos());
        System.out.println("overall mem: " + overallMemoryWatcher.toString());
    }

    public static void addPrediction(Classifier classifier, Instance test, ClassifierResults results) throws Exception {
        final double classValue = test.classValue();
        test.setClassMissing();
        long timestamp = System.nanoTime();
        final double[] distribution = classifier.distributionForInstance(test);
        long testTime = System.nanoTime() - timestamp;
        if(classifier instanceof TestTimeable) {
            testTime = ((TestTimeable) classifier).getTestTime();
        }
        final double prediction = ArrayUtilities.argMax(distribution);
        results.addPrediction(classValue, distribution, prediction, testTime, null);
        test.setClassValue(classValue);
    }
}
