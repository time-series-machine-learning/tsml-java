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
 
package tsml.classifiers.distance_based.utils.collections.iteration;

import java.util.List;
import java.util.ListIterator;

/**
 * Purpose: default implementation of a list iterator.
 *
 * Contributors: goastler
 *
 * @param <A>
 */
public interface DefaultListIterator<A> extends ListIterator<A> {
    default void buildIterator(List<A> list) {
        
    }
    
    @Override
    default boolean hasNext() {
        throw new UnsupportedOperationException();
    }

    @Override
    default A next() {
        throw new UnsupportedOperationException();
    }

    @Override
    default boolean hasPrevious() {
        throw new UnsupportedOperationException();
    }

    @Override
    default A previous() {
        throw new UnsupportedOperationException();
    }

    @Override
    default int nextIndex() {
        throw new UnsupportedOperationException();
    }

    @Override
    default int previousIndex() {
        throw new UnsupportedOperationException();
    }

    @Override
    default void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    default void set(final A a) {
        throw new UnsupportedOperationException();
    }

    @Override
    default void add(final A a) {
        throw new UnsupportedOperationException();
    }
}
