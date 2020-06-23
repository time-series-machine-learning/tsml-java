package tsml.classifiers.distance_based.utils.classifiers;

import tsml.classifiers.EnhancedAbstractClassifier;

public interface EnumBasedClassifierConfigurer<A extends EnhancedAbstractClassifier> extends ClassifierConfigurer<A> {

    @Override default String getClassifierName() {
        // defaults to the name of the enum constant
        return name();
    }

    String name();
}
