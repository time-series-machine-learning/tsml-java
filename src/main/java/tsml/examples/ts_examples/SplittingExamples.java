package tsml.examples.ts_examples;

import java.util.ArrayList;
import java.util.List;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.interval_based.RISE;
import tsml.classifiers.interval_based.TSF;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Splitter;

public class SplittingExamples {
    

    public void example1() throws Exception {
        final double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0, 5.0},
                {4.0,3.0,2.0,1.0, 1.0},
                {4.0,3.0,2.0,1.0, 1.0},
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 1.0},
                {4.0,3.0,2.0,1.0, 1.0},
                {4.0,3.0,2.0,1.0, 1.0},
            }
        };

        final TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[]{0, 1}, new String[]{"A", "B"});

        final List<TimeSeriesInstances> individual_dims = Splitter.splitTimeSeriesInstances(data1);

        //train separate models on univariate data.
        final List<EnhancedAbstractClassifier> clfs = new ArrayList<>(individual_dims.size());
        for(final TimeSeriesInstances data : individual_dims){
            final TSF tsf = new TSF(1);
            tsf.buildClassifier(data);
            clfs.add(tsf);
        }

        //do some ensembling. combine predictions. etc.
    }   


    public void example2() throws Exception {
        final double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0, 5.0},
                {4.0,3.0,2.0,1.0, 1.0},
                {4.0,3.0,2.0,1.0, 1.0},
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 1.0},
                {4.0,3.0,2.0,1.0, 1.0},
                {4.0,3.0,2.0,1.0, 1.0},
            }
        };

        final TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] {"A", "B"});

        final List<TimeSeriesInstances> individual_dims = Splitter.splitTimeSeriesInstances(data1, new int[][]{{0},{1,2}});

        EnhancedAbstractClassifier[] clfs = new EnhancedAbstractClassifier[]{new TSF(), new RISE()};
        for(int i=0; i<individual_dims.size(); i++){
            clfs[i].buildClassifier(individual_dims.get(i));
        }

        //ensemble in some clever way.
    }  

    
}
