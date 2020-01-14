package machine_learning.classifiers.tuned.incremental;

import utilities.ArrayUtilities;
import utilities.collections.PrunedMultimap;

import java.util.*;
import java.util.function.Function;

public class BestBenchmarkCollector implements BenchmarkCollector {
    private final PrunedMultimap<Double, Benchmark> map;
    private int seed;
    private final Random rand = new Random();
    private Function<Benchmark, Double> scorer;
    private Benchmark best = null;

    public BestBenchmarkCollector(Function<Benchmark, Double> scorer) {
        this.map = PrunedMultimap.desc(ArrayList::new);
        map.setSoftLimit(1);
        this.scorer = scorer;
        setSeed(0);
    }

    public BestBenchmarkCollector(int seed, Function<Benchmark, Double> scorer) {
        this(scorer);
        setSeed(seed);
    }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
        map.setSeed(seed);
        rand.setSeed(seed);
    }

    @Override
    public int getSeed() {
        return seed;
    }

    @Override
    public void add(Benchmark benchmark) {
        Double score = scorer.apply(benchmark); // todo add debug to all the interfaces / abst classes
        map.put(score, benchmark);
        best = null;
    }

    @Override
    public Set<Benchmark> getCollectedBenchmarks() {
        if(best == null) {
            Collection<Benchmark> values = map.values();
            List<Benchmark> benchmarks = new ArrayList<>(values);
            best = benchmarks.get(rand.nextInt(benchmarks.size()));
        }
        return new HashSet<>(Collections.singletonList(best));
    }
}
