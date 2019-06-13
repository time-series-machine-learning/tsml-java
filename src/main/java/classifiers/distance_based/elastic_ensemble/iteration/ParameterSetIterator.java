package classifiers.distance_based.elastic_ensemble.iteration;

import evaluation.tuning.ParameterSpace;

import java.util.Iterator;

public class ParameterSetIterator implements Iterator<String[]> {
    private final ParameterSpace parameterSpace;
    private final Iterator<Integer> iterator;

    public ParameterSetIterator(final ParameterSpace parameterSpace,
                                final Iterator<Integer> iterator) {
        this.parameterSpace = parameterSpace;
        this.iterator = iterator;
    }

    @Override
    public void remove() {
        iterator.remove();
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public String[] next() {
        return parameterSpace.get(iterator.next()).getOptions();
    }
}
