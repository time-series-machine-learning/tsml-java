package tsml.classifiers.distance_based.proximity;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.transformers.Transformer;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class VConcatenator implements Transformer {

    public VConcatenator() {
    }

    @Override public Instances determineOutputFormat(final Instances data) throws IllegalArgumentException {
        return data;
    }

    @Override public TimeSeriesInstance transform(final TimeSeriesInstance inst) {

        final List<Double> values = new ArrayList<>();
        for(int i = 0; i < inst.getMaxLength(); i++) {
            for(int j = 0; j < inst.getNumDimensions(); j++) {
                values.add(inst.get(j).get(i));
            }
        }
        
        return new TimeSeriesInstance(Collections.singletonList(values), inst.getLabelIndex());
    }

    @Override public String toString() {
        return "VC";
    }
}
