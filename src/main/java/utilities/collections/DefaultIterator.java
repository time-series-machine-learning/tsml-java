package utilities.collections;

import java.util.Iterator;

public interface DefaultIterator<A> extends Iterator<A> {
    @Override
    default A next() {
        throw new UnsupportedOperationException();
    }

    @Override
    default boolean hasNext() {
        throw new UnsupportedOperationException();
    }

}
