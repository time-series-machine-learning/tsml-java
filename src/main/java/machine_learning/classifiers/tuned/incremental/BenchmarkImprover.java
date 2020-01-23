package machine_learning.classifiers.tuned.incremental;

import java.util.Collection;

public interface BenchmarkImprover extends BenchmarkExplorer {
    default void add(Benchmark benchmark) {
        throw new UnsupportedOperationException();
    }

    default void addAll(Collection<Benchmark> benchmarkCollection) {
        for(Benchmark benchmark : benchmarkCollection) {
            add(benchmark);
        }
    }
}
