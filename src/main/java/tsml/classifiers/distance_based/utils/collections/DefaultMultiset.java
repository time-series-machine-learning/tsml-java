package tsml.classifiers.distance_based.utils.collections;

import com.google.common.collect.Multiset;

import java.util.Collection;
import java.util.Iterator;
import java.util.Set;

public interface DefaultMultiset<K> extends Multiset<K> {
    @Override default int size() {
        throw new UnsupportedOperationException();
    }

    @Override default int count(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override default int add(K k, int i) {
        throw new UnsupportedOperationException();
    }

    @Override default boolean add(K k) {
        throw new UnsupportedOperationException();
    }

    @Override default int remove(Object o, int i) {
        throw new UnsupportedOperationException();
    }

    @Override default boolean remove(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override default int setCount(K k, int i) {
        throw new UnsupportedOperationException();
    }

    @Override default boolean setCount(K k, int i, int i1) {
        throw new UnsupportedOperationException();
    }

    @Override default Set<K> elementSet() {
        throw new UnsupportedOperationException();
    }

    @Override default Set<Entry<K>> entrySet() {
        throw new UnsupportedOperationException();
    }

    @Override default Iterator<K> iterator() {
        throw new UnsupportedOperationException();
    }

    @Override default boolean contains(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override default boolean containsAll(Collection<?> collection) {
        throw new UnsupportedOperationException();
    }

    @Override default boolean removeAll(Collection<?> collection) {
        throw new UnsupportedOperationException();
    }

    @Override default boolean retainAll(Collection<?> collection) {
        throw new UnsupportedOperationException();
    }

    @Override default boolean isEmpty() {
        throw new UnsupportedOperationException();
    }

    @Override default Object[] toArray() {
        throw new UnsupportedOperationException();
    }

    @Override default <T> T[] toArray(T[] ts) {
        throw new UnsupportedOperationException();
    }

    @Override default boolean addAll(Collection<? extends K> collection) {
        throw new UnsupportedOperationException();
    }

    @Override default void clear() {
        throw new UnsupportedOperationException();
    }
}
