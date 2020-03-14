package tsml.classifiers.distance_based.utils.params;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import utilities.ArrayUtilities;

public abstract class ParamDimension {

    private List<ParamSpace> paramsList = new ArrayList<>(); // sub spaces

    /**
     * get the size of each parameter in the space, adding the raw values also.
     * @return
     */
    public final List<Integer> getBins() {
        final List<Integer> subSpaceBins = getSubSpaceBins();
        // bung the dimension size on the end
        subSpaceBins.add(getDimensionSize());
    }

    public abstract int getDimensionSize();

    public final List<Integer> getSubSpaceBins() {
        List<Integer> bins = new ArrayList<>();
        for(ParamSpace paramSets : paramsList) {
            bins.add(paramSets.size());
        }
        return bins;
    }

    public final int size() {
        if(getDimensionSize() < 0) {
            return -1;
        }
        return ArrayUtilities.numPermutations(getBins());
    }

//        public Object abstract

    /**
     * get the index of a value in the space.
     * @param index
     * @return
     */
    public final Object get(final int index) {
        List<Integer> indices = ArrayUtilities.fromPermutation(index, getBins());
        // grab the dimension index off the end
        Object value = values.get(indices.get(indices.size() - 1));
        if(!(value instanceof ParamHandler) && !paramsList.isEmpty()) {
            throw new IllegalStateException("value not param settable");
        }
        for(int i = 0; i < paramsList.size(); i++) {
            ParamHandler paramHandler = (ParamHandler) value;
            ParamSet param = paramsList.get(i).get(indices.get(i));
            paramHandler.setParams(param);
        }
        return value;
    }

    public void addParams(ParamSpace... params) {
        this.paramsList.addAll(Arrays.asList(params));
    }

    public List<?> getValues() {
        return values;
    }

    public void setValues(List<?> values) {
        if(values == null) {
            values = new ArrayList<>();
        }
        this.values = values;
    }

    public List<ParamSpace> getParamsList() {
        return paramsList;
    }

    public void setParamsList(List<ParamSpace> paramsList) {
        if(paramsList == null) {
            paramsList = new ArrayList<>();
        }
        this.paramsList = paramsList;
    }

    @Override
    public String toString() {
        return "ParamValues{" +
            "values=" + values +
            ", params=" + paramsList +
            '}';
    }
}
