package machine_learning.classifiers.tuned.incremental;

import weka.core.Randomizable;

import java.io.Serializable;
import java.util.List;
import java.util.Set;

public interface BenchmarkCollector extends Randomizable, Serializable {
    void add(Benchmark benchmark);
    default void addAll(Iterable<Benchmark> iterable) {
        for(Benchmark benchmark : iterable) {
            add(benchmark);
        }
    }
    Set<Benchmark> getCollectedBenchmarks();
}
