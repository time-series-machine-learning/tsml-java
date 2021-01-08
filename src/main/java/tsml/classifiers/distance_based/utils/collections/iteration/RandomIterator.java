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
import java.util.*;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.system.random.BaseRandom;
import tsml.classifiers.distance_based.utils.system.random.RandomSource;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomIterator<A> extends BaseRandom implements RandomSource, DefaultListIterator<A>, Serializable {

    private Collection<A> collection;
    private int index = -1;
    private boolean nextIndexSetup = false;
    protected boolean withReplacement = true;
    private Iterator<A> iterator;

    public boolean withReplacement() {
        return withReplacement;
    }

    public void setWithReplacement(final boolean withReplacement) {
        this.withReplacement = withReplacement;
    }

    public Collection<A> getCollection() {
        return collection;
    }

    public void setCollection(final Collection<A> collection) {
        Assert.assertNotNull(collection);
        this.collection = collection;
    }

    protected void setRandomIndex() {
        if(!nextIndexSetup) {
            if(collection.isEmpty()) {
                index = -1;
            } else {
                index = getRandom().nextInt(collection.size());
            }
            nextIndexSetup = true;
        }
    }

    public RandomIterator(Random random) {
        this(random, new ArrayList<>(), false);
    }

    public RandomIterator(Random random, Collection<A> collection) {
        this(random, collection, false);
    }

    public RandomIterator(Random random, Collection<A> collection, boolean withReplacement) {
        super(random);
        setCollection(collection);
        setWithReplacement(withReplacement);
    }

    public int nextIndex() {
        setRandomIndex();
        return index;
    }

    public A next() {
        int index = nextIndex();
        nextIndexSetup = false;
        A element = null;
        if(collection instanceof List) {
            element = ((List<A>) collection).get(index);
        } else {
            iterator = collection.iterator();
            for(int i = 0; i <= index; i++) {
                if(!iterator.hasNext()) {
                    throw new IllegalStateException();
                }
                element = iterator.next();
            }
        }
        if(!withReplacement) {
            forceRemove();
        }
        return element;
    }

    @Override
    public boolean hasNext() {
        return !collection.isEmpty();
    }

    private void forceRemove() {
        if(collection instanceof List) {
            ((List<A>) collection).remove(index);
        } else {
            iterator.remove();
        }
    }

    public void remove() {
        if(withReplacement) {
            forceRemove();
        }
    }

    @Override public void set(final A a) {
        if(collection instanceof List) {
            ((List<A>) collection).set(index, a);
        } else {
            throw new UnsupportedOperationException();
        }
    }

    @Override public void add(final A a) {
        collection.add(a);
    }
}
