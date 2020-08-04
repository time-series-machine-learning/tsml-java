package tsml.classifiers.distance_based.utils.classifiers;

import tsml.classifiers.EnhancedAbstractClassifier;

public interface EnumBasedConfigurer<A extends EnhancedAbstractClassifier> extends Configurer<A> {

    String name();

    @Override default <B extends A> B configure(B classifier) {
        classifier = configureFromEnum(classifier);
        classifier.setClassifierName(name());
        return classifier;
    }

    <B extends A> B configureFromEnum(B classifier);
}
