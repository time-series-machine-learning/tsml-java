package classifiers.distance_based.elastic_ensemble.iteration.random;

import classifiers.distance_based.elastic_ensemble.iteration.linear.LinearIterator;

import java.util.List;

public class RoundRobinIterator<A> extends LinearIterator<A> {

    public RoundRobinIterator(final List<A> values) {
        super(values);
    }

    public RoundRobinIterator(RoundRobinIterator<A> other) {
        this(other.values);
        index = other.index;
    }

    public RoundRobinIterator() {
        super();
    }

    @Override
    public A next() {
        return values.get(index = (index + 1) % values.size());
    }

    @Override
    public RoundRobinIterator<A> iterator() {
        return new RoundRobinIterator<>(this);
    }
}
