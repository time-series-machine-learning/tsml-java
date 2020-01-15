package machine_learning.classifiers.tuned.incremental;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.function.Supplier;

public class BenchmarkExplorer implements BenchmarkIterator {
    private BenchmarkImprover benchmarkImprover = new BenchmarkImprover() {
        @Override public Set<Benchmark> getImproveableBenchmarks() {
            throw new UnsupportedOperationException();
        }

        @Override public Set<Benchmark> getUnimprovableBenchmarks() {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean hasNext() {
            throw new UnsupportedOperationException();
        }
    };
    private BenchmarkIterator benchmarkSource = new BenchmarkIterator() {
        // todo bespoke class
        @Override
        public boolean hasNext() {
            return false;
        }

        @Override
        public Set<Benchmark> next() {
            throw new UnsupportedOperationException();
        }
    };
    private Optimiser optimiser = () -> false; // default to fully evaluate each benchmark before sourcing further benchmarks
    private boolean shouldSource = true;
    private int seed = -1;

    @Override
    public boolean hasNext() {
        boolean remainingImprovement = benchmarkImprover.hasNext();
        boolean remainingSource = benchmarkSource.hasNext();
        if(remainingImprovement && remainingSource) {
            // allow the guide to decide whether to improve or source
            shouldSource = optimiser.shouldSource();
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
            benchmarkImprover.addAll(result);
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

    public BenchmarkIterator getBenchmarkSource() {
        return benchmarkSource;
    }

    public void setBenchmarkSource(BenchmarkIterator benchmarkSource) {
        this.benchmarkSource = benchmarkSource;
    }

    public Optimiser getOptimiser() {
        return optimiser;
    }

    public void setOptimiser(Optimiser optimiser) {
        this.optimiser = optimiser;
    }

    @Override public long predictNextTimeNanos() {
        return Math.min(benchmarkSource.predictNextTimeNanos(), benchmarkImprover.predictNextTimeNanos());
    }

}
