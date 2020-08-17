package tsml.classifiers.distance_based.utils.collections.iteration;

import java.io.Serializable;
import java.util.Iterator;

public interface DefaultIterator<A> extends Iterator<A>, Serializable {
    @Override
    default A next() {
        throw new UnsupportedOperationException();
    }

    @Override
    default boolean hasNext() {
        throw new UnsupportedOperationException();
    }

}
