package tsml.classifiers;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.classifiers.Classifier;
import tsml.transformers.Converter;
public interface TSClassifier{
    

    public Classifier getClassifier();

    public default void fit(TimeSeriesInstances data) throws Exception{
        getClassifier().buildClassifier(Converter.toArff(data));
    }

    public default double[] predict_probabilities(TimeSeriesInstance inst) throws Exception{
        return getClassifier().distributionForInstance(Converter.toArff(inst));
    }

    public default double predict(TimeSeriesInstance inst) throws Exception{
        return getClassifier().classifyInstance(Converter.toArff(inst));
    }
}