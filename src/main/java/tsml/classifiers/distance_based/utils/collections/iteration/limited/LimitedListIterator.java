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
 
package tsml.classifiers.distance_based.utils.collections.iteration.limited;

import tsml.classifiers.distance_based.utils.collections.iteration.DefaultListIterator;
import weka.core.OptionHandler;

import java.util.Enumeration;
import java.util.Iterator;

/**
 * Purpose: traverse a list up to a point.
 *
 * Contributors: goastler
 *
 * @param <A>
 */
public class LimitedListIterator<A> implements DefaultListIterator<A>, OptionHandler { // todo abst limit to abst
    // class for iterator and listiterator version

    public LimitedListIterator() {}

    public LimitedListIterator(Iterator<A> iterator) {
        setIterator(iterator);
    }

    public LimitedListIterator(Iterator<A> iterator, int limit) {
        setIterator(iterator);
        setLimit(limit);
    }

    public LimitedListIterator(int limit, Iterator<A> iterator) {
        this(iterator, limit);
    }

    public LimitedListIterator(int limit) {
        setLimit(limit);
    }

    protected int limit = -1;

    public int getLimit() {
        return limit;
    }

    public void setLimit(final int limit) {
        this.limit = limit;
    }

    protected int count = 0;
    protected Iterator<A> iterator;

    public Iterator<A> getIterator() {
        return iterator;
    }

    public void setIterator(final Iterator<A> iterator) {
        this.iterator = iterator;
    }

    @Override
    public boolean hasNext() {
        return (count < limit || limit < 0) && iterator.hasNext();
    }

    @Override
    public A next() {
        count++;
        return iterator.next();
    }

    public void resetCount() {
        count = 0;
    }

    @Override
    public void remove() {
        iterator.remove();
    }

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception { // todo

    }

    @Override
    public String[] getOptions() {
        return new String[] {
            "-l",
            String.valueOf(limit)
        }; // todo
    }

    // todo pass through other iterator funcs

}
