package tsml.classifiers.distance_based.knn.configs;

import tsml.classifiers.distance_based.ee.george_utils.ClassifierBuilderFactory;

public enum KnnTag implements ClassifierBuilderFactory.Tag {
    UNIVARIATE,
    DISTANCE,
    SIMILARITY
    ;

    @Override
    public String getName() {
        return name();
    }

    @Override
    public String toString() {
        return name();
    }
}
