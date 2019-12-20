package machine_learning.classifiers.tuned.incremental;

import utilities.collections.DefaultIterator;
import utilities.iteration.DefaultListIterator;

import java.util.HashSet;
import java.util.Set;

public interface BenchmarkImprover extends DefaultIterator<Set<Benchmark>> {
    default void add(Benchmark benchmark) {}
    default Set<Benchmark> getImproveableBenchmarks() {
        return new HashSet<>();
    }
}
