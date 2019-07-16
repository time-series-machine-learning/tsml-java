package classifiers.template.classifier;

import timeseriesweka.classifiers.CheckpointClassifier;
import timeseriesweka.classifiers.ContractClassifier;
import timeseriesweka.classifiers.SaveParameterInfo;
import utilities.Copyable;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import weka.core.OptionHandler;

import java.io.Serializable;

public interface TemplateClassifierInterface
    extends Serializable,
            SaveParameterInfo,
            CheckpointClassifier,
            ContractClassifier,
            TrainAccuracyEstimate,
            Classifier,
            OptionHandler,
            Copyable {
    Long getTrainSeed();

    void setTrainSeed(Long seed);

    Long getTestSeed();

    void setTestSeed(Long seed);
}
