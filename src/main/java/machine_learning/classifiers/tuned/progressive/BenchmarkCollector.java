package machine_learning.classifiers.tuned.progressive;

import utilities.DefaultRandomizable;
import weka.core.Randomizable;

import java.util.List;

public interface BenchmarkCollector {
    void add(Benchmark benchmark);
    List<Benchmark> getCollectedBenchmarks();
}
