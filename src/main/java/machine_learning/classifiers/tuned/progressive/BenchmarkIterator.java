package machine_learning.classifiers.tuned.progressive;

import machine_learning.classifiers.tuned.progressive.Benchmark;
import utilities.collections.DefaultIterator;

import java.util.List;
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