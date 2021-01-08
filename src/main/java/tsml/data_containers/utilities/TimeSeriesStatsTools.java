/* 
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
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