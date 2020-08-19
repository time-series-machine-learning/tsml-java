package tsml.examples.ts_examples;

import machine_learning.classifiers.kNN;
import tsml.data_containers.TimeSeriesInstances;
import tsml.graphs.Pipeline;
import tsml.transformers.Sine;
import utilities.ClassifierTools;

public class GraphsExample{
    

    public void example1() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 1.0},
            }
        };


        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[]{0, 1});
        data1.setClassLabels(new String[]{"A", "B"});

        double[][][] in1 = {   
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
            }
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[]{0});
        data2.setClassLabels(new String[]{"A", "B"});


        //this is a sequential pipeline.
        Pipeline model = new Pipeline();

        model.add("sine", new Sine());
        model.add("kNN", new kNN());

        model.buildClassifier(data1);

        double acc = ClassifierTools.accuracy(data2, model);
        System.out.println(acc);
    }

    public static void main(String[] args) {
        example1();
    }
}
