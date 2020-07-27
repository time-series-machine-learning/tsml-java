package tsml.transformers;

import tsml.data_containers.*;
import weka.core.Instance;
import weka.core.Instances;

public class Converter implements TrainableTransformer {

    boolean isFit = false;

    @Override
    public Instance transform(Instance inst) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public boolean isFit() {
        return isFit;
    }

    @Override
    public void fit(Instances data) {
        // TODO Auto-generated method stub

    }

    @Override
    public void fit(TimeSeriesInstances data) {
        // TODO Auto-generated method stub

    }

    public static void main(String[] args) {
        
        TimeSeriesInstances insts_ts = null;
        Instances insts_arff = null;

        TrainableTransformer conv = new Converter();

        conv.fit(insts_arff);
        insts_ts = conv.transform(insts_ts);
        
        conv.fit(insts_ts);
        insts_arff = conv.transform(insts_arff);
    }
    
}