package tsml.data_containers.examples;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;

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

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1});
        data.setClassLabels(new String[]{"A", "B"});
        double[] max = new double[data.getMaxLength()];
		double[] min = new double[data.getMaxLength()];
		double[] mean = new double[data.getMaxLength()];
		double[] stdev = new double[data.getMaxLength()];


        //calculate summary stats for each vertical slice of the dataset
		for (int j = 0; j < data.getMaxLength(); j++) {
			double[] slice = data.getSingleSliceArray(j);

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

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[]{0, 1});
        data.setClassLabels(new String[]{"A", "B"});


        //this should produce a size 3 interval slice across all dimensions that include atts: 0,1,2
        double[][][] interval = data.getSliceArray(IntStream.range(0, 3).toArray());
        //equiv: double[][][] interval = data.getSliceArray(new int{0,1,2});


        TimeSeriesInstances data_slice = new TimeSeriesInstances(interval, data.getClassIndexes());
        data_slice.setClassLabels(data.getClassLabels());

        System.out.println("Original");
        System.out.println(data);
        System.out.println("Intveral of 3");
        System.out.println(data_slice);

    }

    public static void main(String[] args) {
        example1();
        example2();
    }
    
}