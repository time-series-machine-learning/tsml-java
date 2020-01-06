package machine_learning.classifiers.tuned.incremental;

import utilities.collections.DefaultIterator;
import utilities.iteration.DefaultListIterator;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public interface BenchmarkImprover extends BenchmarkIterator {
    default void add(Benchmark benchmark) {
        throw new UnsupportedOperationException();
    }
    Set<Benchmark> getImproveableBenchmarks();

    default void addAll(Collection<Benchmark> benchmarkCollection) {
        for(Benchmark benchmark : benchmarkCollection) {
            add(benchmark);
        }
    }

    Set<Benchmark> getUnimprovableBenchmarks();

}
