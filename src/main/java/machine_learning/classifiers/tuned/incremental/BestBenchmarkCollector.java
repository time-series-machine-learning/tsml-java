package machine_learning.classifiers.tuned.incremental;

import utilities.ArrayUtilities;
import utilities.collections.PrunedTreeMultiMap;
import utilities.collections.TreeMultiMap;

import java.util.*;
import java.util.function.Function;

public class BestBenchmarkCollector implements BenchmarkCollector {
    private final PrunedTreeMultiMap<Double, Benchmark> map;
    private int seed = 0;
    private Random rand = new Random(seed);
    private Function<Benchmark, Double> scorer;
    private Benchmark best = null;

    public BestBenchmarkCollector(Function<Benchmark, Double> scorer) {
        this.map = new PrunedTreeMultiMap<Double, Benchmark>(TreeMultiMap.newNaturalAsc());
        map.setLimit(1);
        this.scorer = scorer;
    }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
        rand.setSeed(seed);
    }

    @Override
    public int getSeed() {
        return seed;
    }

    @Override
    public void add(Benchmark benchmark) {
        map.add(scorer.apply(benchmark), benchmark);
        best = null;
    }

    @Override
    public List<Benchmark> getCollectedBenchmarks() {
        if(best == null) {
            List<Benchmark> benchmarks = ArrayUtilities.flatten(map.values()); // todo as view
            best = benchmarks.get(rand.nextInt(benchmarks.size()));
        }
        return new ArrayList<>(Collections.singletonList(best));
    }
}
