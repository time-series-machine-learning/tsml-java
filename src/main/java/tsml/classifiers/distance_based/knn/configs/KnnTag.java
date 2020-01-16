package tsml.classifiers.distance_based.knn.configs;

import java.util.function.Supplier;

public enum KnnTag implements Supplier<String> {
    UNIVARIATE,
    DISTANCE,
    SIMILARITY
    ;

    @Override public String get() {
        return name();
    }
}
