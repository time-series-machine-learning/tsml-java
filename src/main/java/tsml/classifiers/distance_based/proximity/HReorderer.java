package tsml.classifiers.distance_based.proximity;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.transformers.Transformer;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class HReorderer implements Transformer {
    
    public HReorderer() {
        
    }
    
    private List<Integer> indices;
    
    @Override public Instances determineOutputFormat(final Instances data) throws IllegalArgumentException {
        return data;
    }

    @Override public TimeSeriesInstance transform(final TimeSeriesInstance inst) {
        if(indices == null) {
            return inst;
        }
        if(indices.size() != inst.getNumDimensions()) {
            throw new IllegalArgumentException("inst contains " + inst.getNumDimensions() + " dims, only configured to reorder " + indices.size() + " dims");
        }
        
        return inst.getHSlice(indices);
    }

    public List<Integer> getIndices() {
        return indices;
    }

    public void setIndices(final List<Integer> indices) {
        this.indices = indices;
    }
}
