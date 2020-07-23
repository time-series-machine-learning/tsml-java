package tsml.data_containers.utilities;

import java.util.OptionalDouble;

import tsml.data_containers.TimeSeries;

public class TimeSeriesStatsTools {


    public static double mean(TimeSeries ts){
        OptionalDouble out = ts.stream().filter(Double::isFinite).average();
        return out.isPresent() ? out.getAsDouble() : Double.NaN;
    }

    public static TimeSeriesSummaryStatistics getTimeSeriesSummaryStats(TimeSeries ts){
        TimeSeriesSummaryStatistics stats = ts.getSeries().stream().collect(new TimeSeriesCollector());
        return stats;
    }   




    public static void main(String[] args) {
        double [] arr = {1.0, 2.0, Double.NaN, 3.0};
        TimeSeries ts = new TimeSeries(arr);

        double actual = TimeSeriesStatsTools.mean(ts);

        double expected = 2.0;

        System.out.println("Actual " + actual + " expected " + expected);

        TimeSeriesSummaryStatistics stats1 = new TimeSeriesSummaryStatistics(ts.getSeries());

        TimeSeriesSummaryStatistics stats2 = new TimeSeriesSummaryStatistics(ts);

        TimeSeriesSummaryStatistics stats3 = ts.getSeries().stream().collect(new TimeSeriesCollector());

        TimeSeriesSummaryStatistics stats = TimeSeriesStatsTools.getTimeSeriesSummaryStats(ts);
        System.out.println(stats.getMean());
    }


}