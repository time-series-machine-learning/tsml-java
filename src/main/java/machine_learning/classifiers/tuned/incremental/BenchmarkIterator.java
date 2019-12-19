package machine_learning.classifiers.tuned.incremental;

import utilities.collections.DefaultIterator;

import java.util.Set;

public interface BenchmarkIterator extends DefaultIterator<Set<Benchmark>> {
    /**
     * number of benchmarks which will be returned next.
     * @return
     */
    default int nextSize() {
        throw new UnsupportedOperationException();
    }
}
