package classifiers.distance_based.elastic_ensemble;

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
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public String[] next() {
        return parameterSpace.get(iterator.next()).getOptions();
    }
}
