package tsml.classifiers.distance_based.utils.classifiers;

import tsml.classifiers.EnhancedAbstractClassifier;

public interface ClassifierConfigurer<A extends EnhancedAbstractClassifier> extends Configurer<A> {
    @Override default <B extends A> B applyConfigTo(final B classifier) {
        classifier.setClassifierName(getClassifierName());
        return classifier;
    }

    String getClassifierName();
}
