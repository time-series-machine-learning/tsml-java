package classifiers.distance_based.elastic_ensemble.iteration;

import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;

import java.util.Iterator;

public class ParameterSetIterator extends DynamicIterator<ParameterSet, ParameterSetIterator> {
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
    public void add(final ParameterSet parameterSet) {
        parameterSpace.addParameter(parameterSet);
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public ParameterSet next() {
        return parameterSpace.get(iterator.next());
    }

    @Override
    public ParameterSetIterator iterator() {
        return new ParameterSetIterator(parameterSpace, iterator.iterator());
    }
}
