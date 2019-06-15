package classifiers.distance_based.elastic_ensemble.iteration;

import evaluation.tuning.ParameterSpace;

import java.util.Iterator;

public class ParameterSetIterator extends DynamicIterator<String[], ParameterSetIterator> {
    private final ParameterSpace parameterSpace;
    private final DynamicIterator<Integer, ?> iterator;

    public ParameterSetIterator(final ParameterSpace parameterSpace,
                                final DynamicIterator<Integer, ?> iterator) {
        this.parameterSpace = parameterSpace;
        this.iterator = iterator;
    }

    public ParameterSetIterator() {
        throw new UnsupportedOperationException();
    }

    public ParameterSetIterator(ParameterSetIterator other) {
        this(other.parameterSpace, other.iterator); // todo need to copy these!
    }

    @Override
    public void remove() {
        iterator.remove();
    }

    @Override
    public void add(final String[] strings) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public String[] next() {
        return parameterSpace.get(iterator.next()).getOptions();
    }

    @Override
    public ParameterSetIterator iterator() {
        return new ParameterSetIterator(parameterSpace, iterator.iterator());
    }
}
