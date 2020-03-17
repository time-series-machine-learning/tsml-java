package tsml.classifiers.distance_based.utils.params.tmp;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import jdk.nashorn.internal.ir.LiteralNode.ArrayLiteralNode;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.Utilities;

/**
 * hold the parameter dimension. In here should be a method of retreiving values for the given parameter along
 * with sub parameter spaces to explore
 * @param <A>
 */
public class ParameterDimension<A> {

    // list of subspaces to explore
    private List<ParameterSpace> subSpaces;
    // some type holding the values for this dimension. This would usually be a list.
    private A values;

    public ParameterDimension(A values) {
        this(values, new ArrayList<>());
    }

    public ParameterDimension(A values, List<ParameterSpace> subSpaces) {
        setValues(values);
        setSubSpaces(subSpaces);
    }

    @Override
    public String toString() {
        return "{" +
            "values=" + values +
            ", subSpaces=" + subSpaces +
            '}';
    }

    public List<ParameterSpace> getSubSpaces() {
        return subSpaces;
    }

    public void setSubSpaces(final List<ParameterSpace> subSpaces) {
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

    public void addSubSpace(ParameterSpace subSpace) {
        getSubSpaces().add(subSpace);
    }

    public void addValue(A values) {
        this.values = values;
    }

}
