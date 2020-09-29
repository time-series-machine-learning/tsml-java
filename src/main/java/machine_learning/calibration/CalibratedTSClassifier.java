package machine_learning.calibration;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.ClassifierLists;
import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Wrapper class to generically add calibration onto any classifier.
 *
 * Given a classifier and a calibration method (default Dirichlet), will build the classifier and gather train
 * estimates (either via internal mechanisms from the classifier or through an external 3 fold cv), and build the calibrator
 * on these estimates. Predictions of the classifier at test time are then calibrated by the trained calibrator
 */
public class CalibratedTSClassifier extends EnhancedAbstractClassifier {

    //todo review Classifier vs TSClassifier vs EnhancedAbstractClassifier reference decisions
    Classifier classifier;
    Calibrator calibrator;

    ClassifierResults classifierTrainResults;


    public CalibratedTSClassifier(Classifier classifier, Calibrator calibrator) {
        this.classifier = classifier;
        this.calibrator = calibrator;
    }

    public CalibratedTSClassifier(Classifier classifier) {
        this(classifier, new DirichletCalibrator());
    }


    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        buildClassifier(Converter.fromArff(trainData));
    }

    public void buildClassifier(TimeSeriesInstances trainData) throws Exception {
        super.buildClassifier(Converter.toArff(trainData));

        long startTime = System.nanoTime();

        if (EnhancedAbstractClassifier.classifierIsEstimatingOwnPerformance(classifier)) {
            ((EnhancedAbstractClassifier)classifier).buildClassifier(trainData);
            classifierTrainResults = ((EnhancedAbstractClassifier)classifier).getTrainResults();
        }
        else {
            CrossValidationEvaluator cv = new CrossValidationEvaluator();
            cv.setNumFolds(3);
            classifierTrainResults = cv.evaluate(classifier, Converter.toArff(trainData));

            classifier.buildClassifier(Converter.toArff(trainData));
        }

        calibrator.buildCalibrator(classifierTrainResults);
        long calibbuildtime = System.nanoTime() - startTime;

        startTime = System.nanoTime();
        double[][] calibratedProbs = calibrator.calibrateInstances(classifierTrainResults);
        long calibinsttime = (long)((System.nanoTime() - startTime) / (double)trainData.numInstances());

        //this is in fact the train results object made via EnhancedAbstractClassifier
        trainResults = buildCalibratorTrainResults(classifierTrainResults, calibrator.getClass().getSimpleName(),
                calibratedProbs, calibbuildtime, calibinsttime);
    }

    @Override
    public double[] distributionForInstance(TimeSeriesInstance inst) throws Exception {
        double[] classifierProbs = classifier.distributionForInstance(Converter.toArff(inst));
        double[] calibratedProbs = calibrator.calibrateInstance(classifierProbs);
        return calibratedProbs;
    }

    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
        double[] classifierProbs = classifier.distributionForInstance(inst);
        double[] calibratedProbs = calibrator.calibrateInstance(classifierProbs);
        return calibratedProbs;
    }

    protected static ClassifierResults buildCalibratorTrainResults(ClassifierResults classifierTrainResults,
                                                                   String calibratorName, double[][] calibratedProbs, long calibbuildtime, long calibtime) {

        ClassifierResults calibratorResults = new ClassifierResults();
        calibratorResults.setTimeUnit(TimeUnit.NANOSECONDS);
        calibratorResults.setClassifierName(classifierTrainResults.getClassifierName() + "_" + calibratorName);
        calibratorResults.setDatasetName(classifierTrainResults.getDatasetName());
        calibratorResults.setFoldID(classifierTrainResults.getFoldID());
        calibratorResults.setBuildTime(calibbuildtime);

        for (int i = 0; i < classifierTrainResults.numInstances(); i++) {
            double trueval = classifierTrainResults.getTrueClassValue(i);
            double predval = EnhancedAbstractClassifier.findIndexOfMax(calibratedProbs[i], i);
            double[] dist = calibratedProbs[i];
            long predtime = classifierTrainResults.getPredictionTime(i) + calibtime;
            String desc = classifierTrainResults.getPredDescription(i);

            calibratorResults.addPrediction(trueval, dist, predval, predtime, desc);
        }

        return calibratorResults;
    }


    public static void main(String[] args) throws Exception {
        int seed = 0;
        Instances[] data = DatasetLoading.sampleItalyPowerDemand(seed);

        System.out.println(data[0].numInstances());

        Classifier cls = ClassifierLists.setClassifierClassic("RandF", seed);
        CalibratedTSClassifier calibcls = new CalibratedTSClassifier(ClassifierLists.setClassifierClassic("RandF", seed));

        ClassifierResults res = new SingleTestSetEvaluator().evaluate(cls, data[0], data[1]);
        System.out.println(res.allPerformanceMetricsToString());

        System.out.println("\n\n\n\n");
        ClassifierResults res2 = new SingleTestSetEvaluator().evaluate(calibcls, data[0], data[1]);
        System.out.println(res2.allPerformanceMetricsToString());

        System.out.println(Arrays.toString(res2.getPredClassValsAsArray()));
    }
}
