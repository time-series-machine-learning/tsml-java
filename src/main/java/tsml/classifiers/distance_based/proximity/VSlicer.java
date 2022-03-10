package tsml.classifiers.distance_based.proximity;

import tsml.data_containers.TimeSeriesInstance;
import tsml.transformers.Transformer;
import weka.core.Instances;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class VSlicer implements Transformer {

    public VSlicer() {
        this(Collections.emptyList());
    }

    public VSlicer(List<Integer> indices) {
        setIndices(indices);
    }

    private List<Integer> indices;

    @Override public Instances determineOutputFormat(final Instances data) throws IllegalArgumentException {
        return data;
    }

    @Override public TimeSeriesInstance transform(final TimeSeriesInstance inst) {
        return inst.getVSlice(indices);
    }

    public List<Integer> getIndices() {
        return indices;
    }

    public void setIndices(final List<Integer> indices) {
        this.indices = Objects.requireNonNull(indices);
    }
    
    @Override public String toString() {
        return "V" + indices.toString();
    }
}
