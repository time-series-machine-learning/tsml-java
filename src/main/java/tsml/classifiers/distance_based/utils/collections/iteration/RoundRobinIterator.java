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

/**
 * Purpose: round robin traverse a list.
 *
 * Contributors: goastler
 *
 * @param <A>
 */
public class RoundRobinIterator<A> extends LinearIterator<A> {
    public RoundRobinIterator(List<A> list) {
        super(list);
    }

    public RoundRobinIterator() {}

    @Override
    public A next() {
        A next = super.next();
        if(getIndex() == getList().size()) {
            setIndex(0);
        }
        return next;
    }

    @Override
    public void remove() {
        super.remove();
        if(getIndex() < 0) {
            setIndex(getList().size() - 1);
        }
    }

    protected boolean findHasNext() {
        return !getList().isEmpty();
    }

    @Override
    protected int findNextIndex() {
        return super.findNextIndex() % getList().size();
    }
}
