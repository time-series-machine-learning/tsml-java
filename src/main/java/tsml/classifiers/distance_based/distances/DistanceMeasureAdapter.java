package tsml.classifiers.distance_based.distances;

import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

public class DistanceMeasureAdapter implements DistanceMeasure {

    public DistanceMeasureAdapter(final DistanceFunction df) {
        this.df = df;
    }

    private final DistanceFunction df;
    private String[] labels;
    
    private void checkBuilt() {
        if(labels == null) {
            throw new IllegalStateException("distance measure not built - call buildDistanceMeasure or setInstances first");
        }
    }
    
    @Override public double distance(final TimeSeriesInstance a, final TimeSeriesInstance b, final double limit) {
        checkBuilt();
        return distance(Converter.toArff(a, labels), Converter.toArff(b, labels), limit);
    }

    @Override public double distance(final Instance a, final Instance b, final double limit) {
        return df.distance(a, b, limit);
    }

    @Override public ParamSet getParams() {
        final ParamSet paramSet = new ParamSet();
        try {
            paramSet.setOptions(df.getOptions());
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
        return paramSet;
    }

    @Override public void setParams(final ParamSet paramSet) throws Exception {
        df.setOptions(paramSet.getOptions());
    }

    @Override public String getName() {
        return df.getClass().getSimpleName();
    }

    @Override public void buildDistanceMeasure(final TimeSeriesInstances data) {
        buildDistanceMeasure(Converter.toArff(data));
    }

    @Override public void buildDistanceMeasure(final Instances data) {
        df.setInstances(data);
        final Enumeration enumeration = data.classAttribute().enumerateValues();
        final List<String> labels = new ArrayList<>();
        while(enumeration.hasMoreElements()) {
            final Object element = enumeration.nextElement();
            labels.add(element.toString());
        }
        this.labels = labels.toArray(new String[0]);
    }

    @Override public DistanceFunction asDistanceFunction() {
        return df;
    }

    @Override public String toString() {
        return df.toString();
    }
}
