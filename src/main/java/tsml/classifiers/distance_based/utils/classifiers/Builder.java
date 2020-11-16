package tsml.classifiers.distance_based.utils.classifiers;

import java.io.Serializable;

public interface Builder<A> extends Serializable {
    A build();
}
