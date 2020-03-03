package tsml.classifiers.distance_based.knn.configs;

import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory;

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
