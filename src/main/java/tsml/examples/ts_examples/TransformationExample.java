package tsml.examples.ts_examples;

import java.util.function.Function;

import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import tsml.transformers.ShapeletTransform;
import tsml.transformers.shapelet_tools.DefaultShapeletOptions;
import tsml.transformers.shapelet_tools.ShapeletTransformFactory;
import weka.core.Instances;

public class TransformationExample {
    

    public enum MyEnum {

        ONE,TWO;

        public String toString(){
            return this.name();
        }
    }

    public static void main1(){
        MyEnum.ONE.toString();
    }


    // Using a Weka Classifier the annoying way.
    public static void example_full() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                {0.0,1.0,2.0,4.0,5.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 1.0},
                {4.0,3.0,2.0,1.0, 1.0}
            }
        };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[]{0, 1}, new String[]{"A", "B"});

        double[][][] in1 = {   
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                {0.0,1.0,2.0,4.0,5.0}
            }
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[]{0}, new String[]{"A", "B"});

        //build dummy data just to show it works.
        Instances train = Converter.toArff(data1);

        ShapeletTransform trans = new ShapeletTransformFactory(DefaultShapeletOptions.createSHAPELET_D(train))
                                    .getTransform();

        Instances t_train = trans.fitTransform(train);

        System.out.println(t_train);
    }

    // Using a Weka Classifier the annoying way.
    public static void example_timed() throws Exception {
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                {0.0,1.0,2.0,4.0,5.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 1.0},
                {4.0,3.0,2.0,1.0, 1.0}
            }
        };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[]{0, 1}, new String[]{"A", "B"});

        double[][][] in1 = {   
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                {0.0,1.0,2.0,4.0,5.0}
            }
        };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in1, new int[]{0}, new String[]{"A", "B"});

        //build dummy data just to show it works.
        Instances train = Converter.toArff(data1);
        ShapeletTransform trans1 = new ShapeletTransformFactory(DefaultShapeletOptions.createSHAPELET_I_TIMED(train, 10000l, 0))
                                    .getTransform();

        TimeSeriesInstances t_train1 = trans1.fitTransformConverter(train);

        ShapeletTransform trans2 = new ShapeletTransformFactory(DefaultShapeletOptions.createSHAPELET_I_TIMED(train, 10000l, 0))
        .getTransform();

        TimeSeriesInstances t_train2 = trans2.fitTransform(data1);      

        System.out.println(t_train1 == t_train2);
    }

    public static void main(String[] args) throws Exception {
        // example_full();
        example_timed();


        Function<Integer, Function<Integer, Integer>> add = a -> b -> a+b;
        int a = add.apply(10).apply(5);

        Function<Integer, Integer> partial_add = add.apply(10);
        int b = partial_add.apply(5);

    }

}
