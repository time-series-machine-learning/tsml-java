package machine_learning.classifiers.tuned.incremental;

import java.util.Set;

public interface BenchmarkExplorer extends BenchmarkIterator {

    Set<Benchmark> findFinalBenchmarks();

}
