package tsml.classifiers.distance_based.utils.classifiers;

import java.io.Serializable;

public interface Factory<A> extends Serializable {
    A build();
}
