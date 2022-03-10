package tsml.classifiers.distance_based.proximity;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.transformers.Transformer;
import weka.core.Instances;

import java.util.*;

public class HConcatenator implements Transformer {
    
    public HConcatenator() {
    }
    
    @Override public Instances determineOutputFormat(final Instances data) throws IllegalArgumentException {
        return data;
    }

    @Override public TimeSeriesInstance transform(final TimeSeriesInstance inst) {
        
        final List<Double> values = new ArrayList<>();
        for(TimeSeries series : inst) {
            for(Double value : series) {
                values.add(value);
            }
        }
        return new TimeSeriesInstance(Collections.singletonList(values), inst.getLabelIndex());
    }

    @Override public String toString() {
        return "HC";
    }
}
