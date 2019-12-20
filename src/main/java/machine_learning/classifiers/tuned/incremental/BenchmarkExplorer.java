package machine_learning.classifiers.tuned.incremental;

import java.util.Iterator;
import java.util.Set;

public class BenchmarkExplorer implements BenchmarkIterator {
    private BenchmarkImprover benchmarkImprover = new BenchmarkImprover() {
        @Override
        public boolean hasNext() {
            return false;
        }
    };
    private Iterator<Set<Benchmark>> benchmarkSource = new BenchmarkIterator() { // todo bespoke class
        @Override
        public boolean hasNext() {
            return false;
        }

        @Override
        public Set<Benchmark> next() {
            throw new UnsupportedOperationException();
        }
    };
    private Guide guide = () -> false; // default to fully evaluate each benchmark before sourcing further benchmarks
    private boolean shouldSource = true;
//    private Set<Benchmark> improveableBenchmarks = new HashSet<>();

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
        Set<Benchmark> result;
        if(shouldSource) {
            result = benchmarkSource.next();
        } else {
            result = benchmarkImprover.next();
        }
        return result;
    }

    public BenchmarkImprover getBenchmarkImprover() {
        return benchmarkImprover;
    }

    public void setBenchmarkImprover(BenchmarkImprover benchmarkImprover) {
        this.benchmarkImprover = benchmarkImprover;
    }

    public Iterator<Set<Benchmark>> getBenchmarkSource() {
        return benchmarkSource;
    }

    public void setBenchmarkSource(Iterator<Set<Benchmark>> benchmarkSource) {
        this.benchmarkSource = benchmarkSource;
    }

    public Guide getGuide() {
        return guide;
    }

    public void setGuide(Guide guide) {
        this.guide = guide;
    }
}
