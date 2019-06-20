package classifiers.distance_based.elastic_ensemble.iteration.linear;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;

import java.util.*;

public abstract class AbstractLinearIterator<A, B extends AbstractLinearIterator<A, B>> extends DynamicIterator<A, AbstractLinearIterator<A, B>> {
    protected final List<A> values;
    protected int index = 0;

    public AbstractLinearIterator(final Collection<? extends A> values) {
        this.values = new ArrayList<>(values);
    }

    public AbstractLinearIterator() {
        this.values = new ArrayList<>();
    }

    public AbstractLinearIterator(AbstractLinearIterator<A, B> other) {
        this(other.values);
        index = other.index;
    }

    @Override
    public void remove() {
        index--;
        values.remove(index);
        if(index < 0) {
            index = 0;
        }
    }

    @Override
    public void add(final A a) {
        values.add(a);
    }

    @Override
    public boolean hasNext() {
        return !values.isEmpty();
    }

    @Override
    public A next() {
        return values.get(index++);
    }

}
