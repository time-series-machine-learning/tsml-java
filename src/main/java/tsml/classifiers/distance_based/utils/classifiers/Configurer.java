package tsml.classifiers.distance_based.utils.classifiers;

import weka.classifiers.Classifier;

public interface Configurer<A> {

    <B extends A> B configure(B classifier);

    default Configurer<A> and(Configurer<A> other) {
        final Configurer<A> current = this;
        return new Configurer<A>() {
            @Override public <B extends A> B configure(B classifier) {
                classifier = other.configure(classifier);
                classifier = current.configure(classifier);
                return classifier;
            }
        };
    }
}
