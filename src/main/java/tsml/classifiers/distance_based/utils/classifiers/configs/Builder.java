package tsml.classifiers.distance_based.utils.classifiers.configs;

import java.io.Serializable;

public interface Builder<A> extends Serializable {
    A build();
}
