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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

/**
 * Purpose: linearly traverse a list.
 *
 * Contributors: goastler
 *
 * @param <A>
 */
public class LinearListIterator<A> implements ListIterator<A>,
                                              Serializable {

    protected List<A> list = new ArrayList<>();
    protected int index = -1;

    public LinearListIterator(final List<A> list) {
        this.list = list;
    }

    public LinearListIterator() {

    }

    @Override
    public boolean hasNext() {
        return nextIndex() < list.size();
    }

    @Override
    public A next() {
        index = nextIndex();
        return list.get(index);
    }

    @Override
    public boolean hasPrevious() {
        return index >= 0;
    }

    @Override
    public A previous() {
        A previous = list.get(index);
        index = previousIndex();
        return previous;
    }

    @Override
    public int nextIndex() {
        return index + 1;
    }

    @Override
    public int previousIndex() {
        return index - 1;
    }

    @Override
    public void remove() {
        list.remove(index);
        index = previousIndex();
    }

    @Override
    public void set(final A a) {
        list.set(index, a);
    }

    @Override
    public void add(final A a) {
        list.add(a);
    }

    public List<A> getList() {
        return list;
    }

    public void setList(final List<A> list) {
        this.list = list;
    }
}
