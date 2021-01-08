package tsml.examples.ts_examples;

import tsml.data_containers.TSCapabilities;
import tsml.data_containers.TimeSeriesInstances;

public class CapabilitiesExample {
    

    public static void example1() {
        double[][][] in = {
                // instance zero.
                {
                        // time-series zero.
                        { 0.0, 1.0, 2.0, 4.0 }, },
                // instance one
                {
                        // time-series zero.
                        { 4.0, 3.0, 2.0, 1.0 }, } };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[] { "A", "B" });


        TSCapabilities capabilities1 = new TSCapabilities();
        capabilities1.enable(TSCapabilities.EQUAL_LENGTH)
                     .enable(TSCapabilities.UNIVARIATE)
                     .enable(TSCapabilities.NO_MISSING_VALUES);

        boolean canHandle = capabilities1.test(data1);
        System.out.println(canHandle);
    }

    public static void example2() {
        double[][][] in = {
                // instance zero.
                {
                        // time-series zero.
                        { 0.0, 1.0, 2.0, 4.0 }, },
                // instance one
                {
                        // time-series zero.
                        { 4.0, 3.0, 2.0, 1.0 }, } };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[] { "A", "B" });


        TSCapabilities capabilities1 = new TSCapabilities();
        capabilities1.enable(TSCapabilities.EQUAL_LENGTH)
                     .enable(TSCapabilities.UNIVARIATE)
                     .enable(TSCapabilities.NO_MISSING_VALUES)
                     .enable(TSCapabilities.MIN_LENGTH(3));

        boolean canHandle = capabilities1.test(data1);
        System.out.println(canHandle);
    }

    public static void example3() {
        double[][][] in = {
                // instance zero.
                {
                        // time-series zero.
                        { 0.0, 1.0, 2.0, 4.0 }, 
                        { 0.0, 1.0, 2.0 }    
                },
                // instance one
                {
                        // time-series zero.
                        { 4.0, 2.0, 1.0 },
                        { 0.0, 1.0, 2.0, 4.0 }                       
                } 
            };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[] { "A", "B" });


        TSCapabilities capabilities1 = new TSCapabilities();
        capabilities1.enable(TSCapabilities.EQUAL_OR_UNEQUAL_LENGTH)
                     .enable(TSCapabilities.MULTI_OR_UNIVARIATE)
                     .enable(TSCapabilities.NO_MISSING_VALUES);

        boolean canHandle = capabilities1.test(data1);
        System.out.println(canHandle);


        double[][][] in2 = {
            // instance zero.
            {
                    // time-series zero.
                    { 0.0, 1.0, 2.0, 4.0 }, },
            // instance one
            {
                    // time-series zero.
                    { 4.0, 3.0, 2.0, 1.0 }, } };

        TimeSeriesInstances data2 = new TimeSeriesInstances(in2, new int[] { 0, 1 }, new String[] { "A", "B" });

        canHandle = capabilities1.test(data2);
        System.out.println(canHandle);
    }

    public static void example4() {
        double[][][] in = {
                // instance zero.
                {
                        // time-series zero.
                        { 0.0, 1.0, 2.0, 4.0 }, 
                        { 0.0, 1.0, 2.0 }    
                },
                // instance one
                {
                        // time-series zero.
                        { 4.0, 2.0, 1.0 },
                        { 0.0, Double.NaN, 2.0, 4.0 }                       
                } 
            };

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[] { "A", "B" });


        TSCapabilities capabilities1 = new TSCapabilities();
        capabilities1.enable(TSCapabilities.EQUAL_OR_UNEQUAL_LENGTH)
                     .enable(TSCapabilities.MULTIVARIATE)
                     .enable(TSCapabilities.MISSING_VALUES);

        boolean canHandle = capabilities1.test(data1);
        System.out.println(canHandle);
    }


    public static void main(String[] args) {
        example1();
        example2();
        example3();
        example4();
    }
}
