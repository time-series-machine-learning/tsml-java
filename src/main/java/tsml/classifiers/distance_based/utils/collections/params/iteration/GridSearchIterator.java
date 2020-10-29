package tsml.classifiers.distance_based.utils.collections.params.iteration;

import tsml.classifiers.distance_based.utils.collections.iteration.LinearIterator;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete.IndexedParamSpace;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class GridSearchIterator extends ParamSpaceSearch {

    private IndexedParamSpace indexedParamSpace;
    private final LinearIterator<ParamSet> iterator = new LinearIterator<>();

    public GridSearchIterator() {}

    public IndexedParamSpace getIndexedParamSpace() {
        return indexedParamSpace;
    }

    private void setParamSpace(final ParamSpace paramSpace) {
        indexedParamSpace = new IndexedParamSpace(paramSpace);
        iterator.buildIterator(indexedParamSpace);
    }

    @Override public void buildSearch(final ParamSpace paramSpace) {
        super.buildSearch(paramSpace);
        setParamSpace(paramSpace);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "paramSpace=" + indexedParamSpace.getParamSpace().toString() + "}";
    }

    @Override
    public boolean hasNextParamSet() {
        return iterator.hasNext();
    }

    @Override
    public ParamSet nextParamSet() {
        return iterator.next();
    }

    public int size() {
        return indexedParamSpace.size();
    }

}
