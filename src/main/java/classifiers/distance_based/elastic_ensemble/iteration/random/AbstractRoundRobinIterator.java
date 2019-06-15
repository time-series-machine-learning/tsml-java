package classifiers.distance_based.elastic_ensemble.iteration.random;

import classifiers.distance_based.elastic_ensemble.iteration.linear.AbstractLinearIterator;

import java.util.Collection;

public abstract class AbstractRoundRobinIterator<A, B extends AbstractRoundRobinIterator<A, B>> extends AbstractLinearIterator<A, AbstractRoundRobinIterator<A, B>>
     {

    public AbstractRoundRobinIterator(final Collection<? extends A> values) {
        super(values);
    }

    public AbstractRoundRobinIterator(AbstractRoundRobinIterator<A, B> other) {
        this(other.values);
        index = other.index;
    }

    public AbstractRoundRobinIterator() {
        super();
    }

    @Override
    public void remove() {
        values.remove(index--);
    }

    @Override
    public A next() {
        return values.get(index = (index + 1) % values.size());
    }

 }
