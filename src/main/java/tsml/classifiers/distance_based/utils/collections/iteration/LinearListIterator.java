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
