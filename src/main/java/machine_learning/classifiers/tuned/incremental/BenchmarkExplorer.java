package machine_learning.classifiers.tuned.incremental;

import utilities.collections.DefaultIterator;
import weka.core.Randomizable;

import java.util.Set;

public interface BenchmarkExplorer extends DefaultIterator<Set<Benchmark>> {
    default long predictNextTimeNanos() {
        return -1;
    }
    Set<Benchmark> findFinalBenchmarks();
}
