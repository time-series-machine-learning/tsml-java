package tsml.classifiers.distance_based.utils.collections.views;

import tsml.classifiers.distance_based.utils.collections.DefaultList;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;

public class WrappedList<A, B> implements DefaultList<B> {

    public WrappedList(final List<A> list, final Function<A, B> function) {
        this.list = Objects.requireNonNull(list);
        this.function = Objects.requireNonNull(function);
    }

    private final List<A> list;
    private final Function<A, B> function;

    @Override public B get(final int i) {
        return function.apply(list.get(i));
    }

    @Override public int size() {
        return list.size();
    }
}
