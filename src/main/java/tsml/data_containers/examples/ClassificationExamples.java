package tsml.data_containers.examples;

import machine_learning.classifiers.kNN;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.Converter;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class ClassificationExamples {

    // Using a Weka Classifier the annoying way.
    public static void example1() {
        double[][][] in = {
                // instance zero.
                {
                        // time-series zero.
                        { 0.0, 1.0, 2.0, 4.0, 5.0 }, },
                // instance one
                {
                        // time-series zero.
                        { 4.0, 3.0, 2.0, 1.0 }, } };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 });
        data1.setClassLabels(new String[] { "A", "B" });

        double[][][] in1 = { {
                // time-series zero.
                { 0.0, 1.0, 2.0, 4.0, 5.0 }, } };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[] { 0 });
        data2.setClassLabels(new String[] { "A", "B" });

        kNN nn = new kNN(1);
        nn.buildClassifier(Converter.toArff(data1));

        double acc = ClassifierTools.accuracy(Converter.toArff(data2), nn);
        System.out.println(acc);
    }

    // Using a Weka Classifier the annoying way.
    public static void example2() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0},
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

        TSClassifier nn = new TSClassifier(){
            kNN nn = new kNN(1);
			@Override
			public AbstractClassifier getClassifier() {
				return nn;
			}
        };

        nn.fit(data1);

        double acc = ClassifierTools.accuracy(data2, nn);
        System.out.println(acc);
    }

    // Using a Weka Classifier the annoying way.
    public static void example3() throws Exception {
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

        TSClassifier stc = new ShapeletTransformClassifier();
        stc.fit(data1);
        double acc = ClassifierTools.accuracy(data2, stc);
        System.out.println(acc);
    }


    public static void main(String[] args) throws Exception {
        example3();
    }


}