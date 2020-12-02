package tsml.examples.ts_examples;

import java.util.Arrays;
import java.util.stream.IntStream;

import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import weka.core.Instances;

public class DataManipulationExample {

    //Example showing simple vertical slicing.
    public static void example1(){
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0},
                //time-series one.
                {4.0,3.0,2.0,1.0}
            }
        };

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] { "A", "B" });
        
        double[] max = new double[data.getMaxLength()];
		double[] min = new double[data.getMaxLength()];
		double[] mean = new double[data.getMaxLength()];
		double[] stdev = new double[data.getMaxLength()];


        //calculate summary stats for each vertical slice of the dataset
		for (int j = 0; j < data.getMaxLength(); j++) {
			double[] slice = data.getVSliceArray(j);

			max[j] = TimeSeriesSummaryStatistics.max(slice);
			min[j] = TimeSeriesSummaryStatistics.min(slice);
			mean[j] = TimeSeriesSummaryStatistics.mean(slice);
			stdev[j] = Math.sqrt(TimeSeriesSummaryStatistics.variance(slice, mean[j]));
		}
    }

    //Example showing simple vertical slicing.
    public static void example2(){
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0},
                //time-series one.
                {4.0,3.0,2.0,1.0}
            }
        };

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] { "A", "B" });
        


        //this should produce a size 3 interval slice across all dimensions that include atts: 0,1,2
        double[][][] interval = data.getVSliceArray(IntStream.range(0, 3).toArray());
        //equiv: double[][][] interval = data.getSliceArray(new int{0,1,2});


        TimeSeriesInstances data_slice = new TimeSeriesInstances(interval, data.getClassIndexes(), data.getClassLabels());

        System.out.println("Original");
        System.out.println(data);
        System.out.println("Intveral of 3");
        System.out.println(data_slice);

    }

    //truncation example.
    public static void example3(){
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 7.0, 8.0},
                //time-series one.
                {4.0,3.0}
            }
        };

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] {"A", "B"});
        


        //this should produce a size 3 interval slice across all dimensions that include atts: 0,1,2
        double[][][] truncated = data.getVSliceArray(IntStream.range(0, data.getMinLength()).toArray());
        //equiv: double[][][] interval = data.getSliceArray(new int{0,1,2});


        TimeSeriesInstances truncated_data = new TimeSeriesInstances(truncated, data.getClassIndexes(), data.getClassLabels());
        System.out.println("Original");
        System.out.println(data);
        System.out.println("Should be 2 value");
        System.out.println(truncated_data);
    }


    //multivariate unequal example.
    public static void example4(){
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 7.0, 8.0},
                //time-series one.
                {4.0,3.0}
            }
        };

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] { "A", "B" });
        

        Instances converted = Converter.toArff(data);
        System.out.println(converted.toString());
    }


    //univariate example.
    public static void example5(){
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

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] { "A", "B" });
        

        Instances converted = Converter.toArff(data);
        System.out.println(converted.toString());
    }

    //conversion from and backagain. An Weka Instances journey.
    public static void example6(){
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

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] { "A", "B" });
        
        System.out.println(data);

        Instances converted = Converter.toArff(data);
        System.out.println(converted.toString());

        TimeSeriesInstances converted_again = Converter.fromArff(converted);
        System.out.println(converted_again);
    }

    //multivariate unequal example.
    public static void example7(){
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 7.0, 8.0},
                //time-series one.
                {4.0,3.0}
            }
        };

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] { "A", "B" });
        
        System.out.println(data);

        Instances converted = Converter.toArff(data);
        System.out.println(converted.toString());

        TimeSeriesInstances converted_again = Converter.fromArff(converted);
        System.out.println(converted_again);
    }

    //HSlicing example.
    public static void example8(){
        double[][][] in = {
            //instance zero.
            {
                //time-series zero.
                {0.0,1.0,2.0,4.0,5.0},
                //time-series one.
                {0.0,1.0,2.0,4.0}
            },
            //instance one
            {
                //time-series zero.
                {4.0,3.0,2.0,1.0, 7.0, 8.0},
                //time-series one.
                {4.0,3.0}
            }
        };

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1}, new String[] { "A", "B" });
        


        //this should produce only dimension 
        double[][][] single_dimension = data.getHSliceArray(IntStream.range(0, 1).toArray());
        //equiv: double[][][] single_dimension = data.getHSliceArray(new int{0});

        System.out.println(Arrays.deepToString(single_dimension));


        TimeSeriesInstances truncated_data = new TimeSeriesInstances(single_dimension, data.getClassIndexes(), data.getClassLabels());

        System.out.println("Original");
        System.out.println(data);
        System.out.println("Should be 2 value");
        System.out.println(truncated_data);
    }

    public static void main(String[] args) {
        //example1();
        //example2();
        //example3();
        //example4();
        // example5();
        // example6();
        //example7();
        example8();
    }
    
}
