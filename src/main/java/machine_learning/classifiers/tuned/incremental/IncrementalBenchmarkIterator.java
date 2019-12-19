package machine_learning.classifiers.tuned.incremental;

import java.util.Set;

public class IncrementalBenchmarkIterator implements BenchmarkIterator {
    private BenchmarkIterator benchmarkImprover = new BenchmarkIterator() {};
    private BenchmarkIterator benchmarkSource = new BenchmarkIterator() {};
    private Guide guide = () -> false; // default to fully evaluate each benchmark before sourcing further benchmarks
    private boolean shouldSource = true;

    public interface Guide {
        boolean shouldSource();
    }

    @Override
    public boolean hasNext() {
        boolean remainingImprovement = benchmarkImprover.hasNext();
        boolean remainingSource = benchmarkSource.hasNext();
        if(remainingImprovement && remainingSource) {
            // allow the guide to decide whether to improve or source
            shouldSource = guide.shouldSource();
        } else if(remainingImprovement) {
            // no remaining source so must improve
            shouldSource = false;
        } else if(remainingSource) {
            // no remaining improvements so must source
            shouldSource = true;
        } else {
            // neither improvements or further benchmarks remain
            return false;
        }
        return true;
    }

    @Override
    public Set<Benchmark> next() {
        if(shouldSource) {
            return benchmarkSource.next();
        } else {
            return benchmarkImprover.next();
        }
    }

    public BenchmarkIterator getBenchmarkImprover() {
        return benchmarkImprover;
    }

    public void setBenchmarkImprover(BenchmarkIterator benchmarkImprover) {
        this.benchmarkImprover = benchmarkImprover;
    }

    public BenchmarkIterator getBenchmarkSource() {
        return benchmarkSource;
    }

    public void setBenchmarkSource(BenchmarkIterator benchmarkSource) {
        this.benchmarkSource = benchmarkSource;
    }
}
