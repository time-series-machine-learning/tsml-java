package utilities.iteration.linear;

import java.util.List;

public class RoundRobinIterator<A> extends LinearIterator<A> {

    public RoundRobinIterator(final List<A> values) {
        super(values);
        index = values.size() - 1;
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
        index = (index + 1) % values.size();
        A value = values.get(index);
        return value;
    }

    @Override
    public void remove() {
        values.remove(index);
        if(index < 0) {
            index = values.size() - 1;
        }
    }

    @Override
    public RoundRobinIterator<A> iterator() {
        return new RoundRobinIterator<>(this);
    }
}
