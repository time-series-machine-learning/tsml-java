package machine_learning.classifiers.tuned.incremental;

import weka.core.Randomizable;

import java.util.List;
import java.util.Set;

public interface BenchmarkCollector extends Randomizable {
    void add(Benchmark benchmark);
    Set<Benchmark> getCollectedBenchmarks();
}
