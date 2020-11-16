package tsml.classifiers.distance_based.utils.classifiers;

import tsml.classifiers.EnhancedAbstractClassifier;

import java.util.Objects;

public interface ClassifierFromEnum<A extends EnhancedAbstractClassifier> extends Configurer<A>, Builder<A> {

    @Override <B extends A> B configure(B classifier);

    @Override default A build() {
        return configure(newInstance());
    }
    
    A newInstance();
}
