package tsml.classifiers.distance_based.knn;

import weka.core.Instances;

import java.util.Iterator;
import java.util.List;

public interface TuningAgent extends Iterator<Benchmark> {

    /**
     * feedback an improved benchmark
     * @param benchmark
     * @return true if the benchmark cannot be improved further / will not be seen again, false otherwise
     */
    boolean feedback(Benchmark benchmark);

    List<Benchmark> getAllBenchmarks();

    default void buildAgent(Instances trainData) {};
}
