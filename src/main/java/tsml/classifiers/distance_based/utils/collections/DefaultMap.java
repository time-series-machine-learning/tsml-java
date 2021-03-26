package tsml.classifiers.distance_based.utils.collections;

import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;

public interface DefaultMap<A, B> extends Map<A, B> {

    @Override default int size() {
        throw new UnsupportedOperationException();
    }

    @Override default boolean isEmpty() {
        return size() == 0;
    }

    @Override default boolean containsKey(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override default boolean containsValue(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override default B get(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override default B put(A a, B b) {
        throw new UnsupportedOperationException();
    }

    @Override default B remove(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override default void putAll(Map<? extends A, ? extends B> map) {
        for(Map.Entry<? extends A, ? extends B> entry : map.entrySet()) {
            put(entry.getKey(), entry.getValue());
        }
    }

    @Override default void clear() {
        throw new UnsupportedOperationException();
    }

    @Override default Set<A> keySet() {
        throw new UnsupportedOperationException();
    }

    @Override default Collection<B> values() {
        throw new UnsupportedOperationException();
    }

    @Override default Set<Entry<A, B>> entrySet() {
        throw new UnsupportedOperationException();
    }

}
