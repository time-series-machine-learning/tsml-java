package machine_learning.calibration;

import tsml.classifiers.TSClassifier;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.EnsureNonZero;
import tsml.transformers.Log;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;

public class DirichletCalibrator  implements Calibrator {

    Logistic regressor;
    TSClassifier tsregressor;

    @Override
    public void buildCalibrator(TimeSeriesInstances classifierProbs) throws Exception {
        regressor = new Logistic();
        tsregressor = new TSClassifier() {
            @Override
            public Classifier getClassifier() {
                return regressor;
            }
        };

        classifierProbs = new EnsureNonZero().transform(classifierProbs);
        classifierProbs = new Log().transform(classifierProbs);
        tsregressor.buildClassifier(classifierProbs);
    }

    @Override
    public double[] calibrateInstance(TimeSeriesInstance classifierProbs) throws Exception {
        classifierProbs = new EnsureNonZero().transform(classifierProbs);
        classifierProbs = new Log().transform(classifierProbs);
        return tsregressor.distributionForInstance(classifierProbs);
    }

}
