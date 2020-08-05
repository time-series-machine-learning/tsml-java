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