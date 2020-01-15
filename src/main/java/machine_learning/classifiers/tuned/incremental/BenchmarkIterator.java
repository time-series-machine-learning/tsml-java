package machine_learning.classifiers.tuned.incremental;

import utilities.collections.DefaultIterator;
import weka.core.Randomizable;

import java.util.Iterator;
import java.util.Set;

public interface BenchmarkIterator extends DefaultIterator<Set<Benchmark>> {
    default long predictNextTimeNanos() {
        return -1;
    }
}
