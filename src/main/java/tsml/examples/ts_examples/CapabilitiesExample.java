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

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 });
        data1.setClassLabels(new String[] { "A", "B" });


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

        TimeSeriesInstances data1 = new TimeSeriesInstances(in, new int[] { 0, 1 });
        data1.setClassLabels(new String[] { "A", "B" });


        TSCapabilities capabilities1 = new TSCapabilities();
        capabilities1.enable(TSCapabilities.EQUAL_LENGTH)
                     .enable(TSCapabilities.UNIVARIATE)
                     .enable(TSCapabilities.NO_MISSING_VALUES)
                     .enable(TSCapabilities.MIN_LENGTH(3));

        boolean canHandle = capabilities1.test(data1);
        System.out.println(canHandle);
    }


    public static void main(String[] args) {
        example1();
        example2();
    }
}