package classifiers.distance_based.elastic_ensemble.iteration;

import java.util.Collection;
import java.util.Iterator;
import java.util.ListIterator;

public abstract class DynamicIterator<A, B extends DynamicIterator<A, B>>
    implements ListIterator<A>, Iterable<A> {


    @Override
    public boolean hasPrevious() {
        return false;
    }

    @Override
    public A previous() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int nextIndex() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int previousIndex() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void set(final A a) {
        throw new UnsupportedOperationException();
    }

    @Override
    public abstract B iterator();

    public void addAll(Collection<A> collection) {
        for(A item : collection) {
            add(item);
        }
    }
}
