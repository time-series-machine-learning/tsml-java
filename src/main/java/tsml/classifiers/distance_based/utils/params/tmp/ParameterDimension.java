package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.ArrayList;
import java.util.List;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.Utilities;

/**
 * hold the parameter dimension. In here should be a method of retreiving values for the given parameter along
 * with sub parameter spaces to explore
 * @param <A>
 */
public abstract class ParameterDimension<A> {

    // list of subspaces to explore
    private List<ParameterSpace> subSpaces = new ArrayList<>();

    public abstract String toString();

    public String buildSubSpacesString() {
        final List<ParameterSpace> subSpaces = getSubSpaces();
        if(subSpaces.isEmpty()) {
            return "";
        }
        return ", subSpaces=" + subSpaces;
    }

    /**
     * some method of retreiving a parameter value (given an index, which may or may not be used as is context
     * dependent)
     * @param index
     * @return
     */
    public abstract A getParameterValue(int index);

    /**
     * get the number of possible values for this parameter NOT including the sub spaces!
     * @return <0 means an infinite number of values, >=0 is the finite size. An infinite size indicates the
     * parameter is continuous, finite indicates it is an enumerated set of values.
     */
    public abstract int getParameterDimensionSize();

    public List<ParameterSpace> getSubSpaces() {
        return subSpaces;
    }

    public void setSubSpaces(final List<ParameterSpace> subSpaces) {
        Assert.assertNotNull(subSpaces);
        this.subSpaces = subSpaces;
    }

    public void addSubSpace(ParameterSpace subSpace) {
        getSubSpaces().add(subSpace);
    }

    public boolean containsContinuousDimension() {
        if(getParameterDimensionSize() < 0) {
            return true;
        }
        List<Integer> sizes = getSubDimensionSizes();
        for(Integer size : sizes) {
            if(size < 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * the size of the parameter dimension. Note this is only the FINITE size! If the parameter is continuous
     * then there is no concept of size. A continuous parameter may have subspaces which are discrete and finite
     * however. For example, I may have a discrete parameter set of [dtw, ddtw]. I may then have a subspace which
     * is continuous, say [warpingWindow{0 : 1}]. This method would return 2, reflecting the two discrete options
     * (dtw and ddtw). It does not take into account the infinite range of the subspace parameter warpingWindow.
     * @return
     */
    public int size() { // todo make sure all of this works with index / size <0 or =0
        final List<Integer> sizes = getSubDimensionSizes();
        // put this parameter's dimension size on the front of the sub sizes
        sizes.add(0, getParameterDimensionSize());
        return Permutations.numPermutations(sizes);
    }

    public List<Integer> getSubDimensionSizes() {
        return Utilities.convert(getSubSpaces(), ParameterSpace::size);
    }

    /**
     * get a ParamSet using a given index. The index is only used to index the finite permutations of parameters
     * where their values have been enumerated. Infinite / continuous parameters are sampled using a pdf and have
     * no bearing on the index variable. If you are only using pdf parameters then the index variable is ignored.
     * @param index
     * @return
     */
    public A get(int index) {
        final List<Integer> sizes = getSubDimensionSizes();
        // find the indices of the dimension and each sub space using the permutation index
        final List<Integer> indices = Permutations.fromPermutation(index, sizes);
        // get this parameter's value index, leaving only sub space indices
        final int valueIndex = indices.remove(0);
        final A value = getParameterValue(valueIndex);
        final List<ParameterSpace> subSpaces = getSubSpaces();
        Assert.assertEquals(subSpaces.size(), indices.size());
        // if we've got any sub spaces
        if(!sizes.isEmpty()) {
            // check if we can set params
            if(!(value instanceof ParamHandler)) {
                throw new IllegalStateException("value not param settable");
            } else {
                // set parameter for each sub space using the indices
                ParamHandler paramHandler = (ParamHandler) value;
                for(int i = 0; i < subSpaces.size(); i++) {
                    int subSpaceValueIndex = indices.get(i);
                    ParameterSpace subSpace = subSpaces.get(i);
                    if(subSpaceValueIndex >= 0 || subSpace.containsContinuousDimension()) {
                        ParamSet param = subSpace.get(subSpaceValueIndex);
                        paramHandler.setParams(param);
                    }
                }
            }
        }
        return value;
    }

}
