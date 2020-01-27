package tsml.classifiers.distance_based.knn.configs;

import experiments.ClassifierBuilderFactory;

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
