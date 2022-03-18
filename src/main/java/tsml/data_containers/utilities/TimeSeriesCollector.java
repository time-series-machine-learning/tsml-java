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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TimeSeriesCollector implements Collector<Double, List<Double>, TimeSeriesSummaryStatistics> {

    @Override
    public BiConsumer<List<Double>, Double> accumulator() {
        return (list, val) -> list.add(val);
    }

    @Override
    public Set<Characteristics> characteristics() {
        HashSet<Characteristics> set = new HashSet<Characteristics>(); 
        set.add(Characteristics.UNORDERED);
        return set;
    }

    //merge two lists in parallel.
    @Override
    public BinaryOperator<List<Double>> combiner() {
       return (list1, list2) -> Stream.concat(list1.stream(), list2.stream()).collect(Collectors.toList());
    }

    @Override
    public Supplier<List<Double>> supplier() {
        return ArrayList<Double>::new;
    }

    @Override
    public Function<List<Double>, TimeSeriesSummaryStatistics> finisher() {
        return TimeSeriesSummaryStatistics::new;
    }
    
}