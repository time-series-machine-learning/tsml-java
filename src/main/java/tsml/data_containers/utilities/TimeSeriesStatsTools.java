package tsml.data_containers.utilities;

import static org.junit.Assert.assertSame;

import java.util.Arrays;
import java.util.OptionalDouble;

import tsml.data_containers.TimeSeries;

public class TimeSeriesStatsTools {


    public static double mean(TimeSeries ts){
        OptionalDouble out = Arrays.stream(ts.getSeries()).filter(Double::isFinite).average();
        return out.isPresent() ? out.getAsDouble() : Double.NaN;
    }

    public static void main(String[] args) {
        double [] arr = {1.0, 2.0, Double.NaN, 3.0};
        TimeSeries ts = new TimeSeries(arr);

        double actual = TimeSeriesStatsTools.mean(ts);

        double expected = 2.0;

        System.out.println("Actual " + actual + " expected " + expected);
    }


}