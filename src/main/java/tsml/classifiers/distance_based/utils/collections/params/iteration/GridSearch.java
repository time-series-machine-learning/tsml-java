package tsml.classifiers.distance_based.utils.collections.params.iteration;

import tsml.classifiers.distance_based.utils.collections.iteration.LinearIterator;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete.GridParamSpace;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class GridSearch extends AbstractSearch {

    private GridParamSpace gridParamSpace;
    private final LinearIterator<ParamSet> iterator = new LinearIterator<>();

    public GridSearch() {}

    public GridParamSpace getIndexedParamSpace() {
        return gridParamSpace;
    }

    private void setParamSpace(final ParamSpace paramSpace) {
        gridParamSpace = new GridParamSpace(paramSpace);
        iterator.buildIterator(gridParamSpace);
    }

    @Override public void buildSearch(final ParamSpace paramSpace) {
        super.buildSearch(paramSpace);
        setParamSpace(paramSpace);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "paramSpace=" + gridParamSpace.getParamSpace().toString() + "}";
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
        return gridParamSpace.size();
    }

}
