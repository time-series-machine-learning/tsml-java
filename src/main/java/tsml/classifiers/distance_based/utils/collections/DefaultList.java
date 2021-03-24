/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.collections;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

public interface DefaultList<A>
    extends List<A> {

    @Override default int size() {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default boolean isEmpty() {
        int size = size();
        return size == 0;
    }

    @Override default boolean contains(final Object o) {

        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default Iterator<A> iterator() {
        return new Iterator<A>() {
            
            private int i = 0;
            private int size = size();
            
            @Override public boolean hasNext() {
                return i < size;
            }

            @Override public A next() {
                return get(i++);
            }
        };
    }

    @Override default Object[] toArray() {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default <T> T[] toArray(final T[] ts) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default boolean add(final A a) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default boolean remove(final Object o) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default boolean containsAll(final Collection<?> collection) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default boolean addAll(final Collection<? extends A> collection) {
        for(A item : collection) {
            add(item);
        }
        return true;
    }

    @Override default boolean addAll(final int i, final Collection<? extends A> collection) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default boolean removeAll(final Collection<?> collection) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default boolean retainAll(final Collection<?> collection) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default void clear() {

        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default A get(final int i) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default A set(final int i, final A a) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default void add(final int i, final A a) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default A remove(final int i) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default int indexOf(final Object o) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default int lastIndexOf(final Object o) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default ListIterator<A> listIterator() {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default ListIterator<A> listIterator(final int i) {
        throw new UnsupportedOperationException("default method not implemented");
    }

    @Override default List<A> subList(final int i, final int i1) {
        throw new UnsupportedOperationException("default method not implemented");
    }
}
