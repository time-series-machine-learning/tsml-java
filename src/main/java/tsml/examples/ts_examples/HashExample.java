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
 
package tsml.examples.ts_examples;

import java.util.Arrays;
import java.util.List;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

public class HashExample {
 
    
    public static void example1() {
        double[] in = { 0.0, 1.0, 2.0, 4.0, 5.0 };
        List<Double> in_list = Arrays.asList(0.0, 1.0, 2.0, 4.0, 5.0);

        TimeSeries data = new TimeSeries(in);

        System.out.println(data.hashCode() == Arrays.hashCode(in));
        System.out.println(data.hashCode() == in_list.hashCode());
    }

    public static void example2() {
        double[][] in = { 
                            {0.0, 1.0, 2.0, 4.0, 5.0} 
                        };

        List<List<Double>> in_list = Arrays.asList(Arrays.asList(0.0, 1.0, 2.0, 4.0, 5.0));

        TimeSeriesInstance data1 = new TimeSeriesInstance(in, 0, new String[]{"A", "B"});

        System.out.println(data1.hashCode() == Arrays.deepHashCode(in));
        System.out.println(data1.hashCode() == in_list.hashCode());
    }

    public static void example3() {
        double[][][] in = {
                {
                        { 0.0, 1.0, 2.0, 4.0, 5.0 }, },
                {
                        { 4.0, 3.0, 2.0, 1.0 }, 
                } 
            };

        List<List<List<Double>>> in_list = Arrays.asList(
                                                Arrays.asList(Arrays.asList(0.0, 1.0, 2.0, 4.0, 5.0)),
                                                Arrays.asList(Arrays.asList(4.0, 3.0, 2.0, 1.0))
                                            );

        TimeSeriesInstances data = new TimeSeriesInstances(in, new int[] { 0, 1 }, new String[]{"A", "B"});


        System.out.println(data.hashCode() == Arrays.deepHashCode(in));
        System.out.println(data.hashCode() == in_list.hashCode());
    }

    public static void main(String[] args) {
        example1();
        example2();
        example3();
    }
}
