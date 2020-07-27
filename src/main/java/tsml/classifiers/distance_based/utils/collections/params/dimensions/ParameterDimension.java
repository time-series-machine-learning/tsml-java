package tsml.classifiers.distance_based.utils.collections.params.dimensions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;

/**
 * hold the parameter dimension. In here should be a method of retreiving values for the given parameter along
 * with sub parameter spaces to explore
 * @param <A>
 */
public abstract class ParameterDimension<A> implements Serializable {

    // list of subspaces to explore
    private List<ParamSpace> subSpaces;
    // some type holding the values for this dimension. This would usually be a list.
    private A values;

    public ParameterDimension(A values) {
        this(values, new ArrayList<>());
    }

    public ParameterDimension(A values, List<ParamSpace> subSpaces) {
        setValues(values);
        setSubSpaces(subSpaces);
    }

    @Override
    public String toString() {
        String subSpacesString = "";
        if(!getSubSpaces().isEmpty()) {
            subSpacesString = ", subSpaces=" + subSpaces;
        }
        return "{" +
            "values=" + values +
            subSpacesString +
            '}';
    }

    public List<ParamSpace> getSubSpaces() {
        return subSpaces;
    }

    public void setSubSpaces(final List<ParamSpace> subSpaces) {
        Assert.assertNotNull(subSpaces);
        this.subSpaces = subSpaces;
    }


    public A getValues() {
        return values;
    }

    public ParameterDimension<A> setValues(final A values) {
        Assert.assertNotNull(values);
        this.values = values;
        return this;
    }

    public void addSubSpace(ParamSpace subSpace) {
        getSubSpaces().add(subSpace);
    }

    public void addValue(A values) {
        this.values = values;
    }

}
