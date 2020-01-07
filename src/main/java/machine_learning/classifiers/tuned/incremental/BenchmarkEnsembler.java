package machine_learning.classifiers.tuned.incremental;

import utilities.collections.Utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public interface BenchmarkEnsembler {
    List<Double> weightVotes(List<Benchmark> benchmarks);

    static BenchmarkEnsembler byScore(Function<Benchmark, Double> scorer) {
        return (benchmarks) -> {
            List<Double> weights = new ArrayList<>();
            for(Benchmark benchmark : benchmarks) {
                weights.add(scorer.apply(benchmark));
            }
            return weights;
        };
    }

    static BenchmarkEnsembler single() {
        return (benchmarks) -> {
            if(Utils.size(benchmarks) != 1) {
                throw new IllegalArgumentException("was only expecting 1 benchmark");
            }
            return new ArrayList<>(Collections.singletonList(1.0));
        };
    }
}
