package tsml.classifiers.distance_based.utils.classifiers;

import java.io.Serializable;

public interface Configurer<A> {
    <B extends A> B applyConfigTo(B classifier);
}
