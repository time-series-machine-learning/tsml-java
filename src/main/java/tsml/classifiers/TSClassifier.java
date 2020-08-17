package tsml.classifiers;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.classifiers.Classifier;
public interface TSClassifier{
    

    public Classifier getClassifier();

    public default void buildClassifier(TimeSeriesInstances data) throws Exception{
        getClassifier().buildClassifier(Converter.toArff(data));
    }

    public default double[] distributionForInstance(TimeSeriesInstance inst) throws Exception{
        return getClassifier().distributionForInstance(Converter.toArff(inst));
    }

    public default double classifyInstance(TimeSeriesInstance inst) throws Exception{
        return getClassifier().classifyInstance(Converter.toArff(inst));
    }

    public default double[][] distributionForInstances(TimeSeriesInstances data) throws Exception {
        double[][] out = new double[data.numInstances()][];
        int i=0;
        for(TimeSeriesInstance inst : data)
            out[i++] = getClassifier().distributionForInstance(Converter.toArff(inst));
        return out;
    }

    public default double[] classifyInstances(TimeSeriesInstances data) throws Exception {
        double[] out = new double[data.numInstances()];
        int i=0;
        for(TimeSeriesInstance inst : data)
            out[i++] = getClassifier().classifyInstance(Converter.toArff(inst));
        return out;
    }
}