package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.ArrayList;
import java.util.List;
import org.junit.Assert;

/**
 * class to hold a finite set of values for a parameter
 * @param <A>
 */
public class DiscreteParameterDimension<A> extends ParameterDimension<A> {

    private List<? extends A> values;

    public DiscreteParameterDimension() {
        this(new ArrayList<>());
    }

    public DiscreteParameterDimension(List<? extends A> values) {
        this(values, new ArrayList<>());
    }

    public DiscreteParameterDimension(List<? extends A> values, List<ParameterSpace> subSpaces) {
        setValues(values);
        setSubSpaces(subSpaces);
    }

    @Override
    public A getParameterValue(final int index) {
        // simply grab the value at the given index
        return values.get(index);
    }

    @Override
    public int getParameterDimensionSize() {
        return values.size();
    }

    public List<? extends A> getValues() {
        return values;
    }

    public void setValues(final List<? extends A> values) {
        Assert.assertNotNull(values);
        this.values = values;
    }

    @Override
    public String toString() {
        return "{values=" + values + super.buildSubSpacesString() + "}";
    }
}
