package machine_learning.classifiers.tuned.incremental;

import java.util.List;

public interface BenchmarkCollector {
    void add(Benchmark benchmark);
    List<Benchmark> getCollectedBenchmarks();
}
