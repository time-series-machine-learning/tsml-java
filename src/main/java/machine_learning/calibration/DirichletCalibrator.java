package machine_learning.calibration;

import tsml.classifiers.TSClassifier;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.EnsureNonZero;
import tsml.transformers.Log;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

/**
 * Starter implementation of Dirichlet calibration. Does not include implementation speed up details described in the paper,
 * nor as yet options for regularisation methods. This should have little effect on datasets with many train instances,
 * however would have a potentially large effect on smaller datasets, and not be representative of the literature-version method.
 *
 * TODO: implement/include packages to improve the logistic regression capabliities, regularisation and parameterisation
 * TODO: decide on and fill out tsml software engineering
 * TODO: finish conversion to TimeSeriesInstances only once codebase-wide branch pushed
 *
 * @inproceedings{kull2019beyond,
 *   title={Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration},
 *   author={Kull, Meelis and Nieto, Miquel Perello and K{\"a}ngsepp, Markus and Silva Filho, Telmo and Song, Hao and Flach, Peter},
 *   booktitle={Advances in Neural Information Processing Systems},
 *   pages={12316--12326},
 *   year={2019}
 * }
 */
public class DirichletCalibrator  implements Calibrator, TechnicalInformationHandler {

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

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.INPROCEEDINGS);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Kull, Meelis and Nieto, Miquel Perello and K{\\\"a}ngsepp, Markus and Silva Filho, Telmo and Song, Hao and Flach, Peter");
        result.setValue(TechnicalInformation.Field.TITLE, "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Advances in Neural Information Processing Systems");
        result.setValue(TechnicalInformation.Field.PAGES, "12316--12326");
        result.setValue(TechnicalInformation.Field.YEAR, "2019");
        return result;
    }
}
