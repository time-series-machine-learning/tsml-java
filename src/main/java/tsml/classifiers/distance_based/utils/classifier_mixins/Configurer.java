package tsml.classifiers.distance_based.utils.classifier_mixins;

import java.io.Serializable;

public interface Configurer<A> {
    <B extends A> B applyConfigTo(B classifier);
}
