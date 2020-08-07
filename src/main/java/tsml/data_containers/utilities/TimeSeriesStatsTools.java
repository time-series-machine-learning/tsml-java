package tsml.data_containers.utilities;

import tsml.data_containers.TimeSeries;

public class TimeSeriesStatsTools {

    
    /** 
     * @param ts
     * @return double
     */
    public static double mean(TimeSeries ts){
        return TimeSeriesSummaryStatistics.mean(ts);
    }

    
    /** 
     * @param ts
     * @return double
     */
    public static double std(TimeSeries ts){
        double mean = TimeSeriesSummaryStatistics.mean(ts);
        return Math.sqrt(TimeSeriesSummaryStatistics.variance(ts, mean));
    }

    
    /** 
     * @param ts
     * @return TimeSeriesSummaryStatistics
     */
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