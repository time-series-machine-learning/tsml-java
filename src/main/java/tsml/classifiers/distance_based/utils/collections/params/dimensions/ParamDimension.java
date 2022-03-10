package tsml.classifiers.distance_based.utils.collections.params.dimensions;

import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * hold the parameter dimension. In here should be a method of retreiving values for the given parameter along
 * with sub parameter spaces to explore
 * @param <A>
 */
public abstract class ParamDimension<A> implements Serializable {

    // list of subspaces to explore
    private ParamSpace subSpace;

    public ParamDimension() {
        this(new ParamSpace());
    }

    public ParamDimension(ParamSpace subSpace) {
        setSubSpace(subSpace);
    }

    @Override
    public String toString() {
        if(!getSubSpace().isEmpty()) {
            return ", subSpace=" + subSpace;
        }
        return "";
    }

    public ParamSpace getSubSpace() {
        return subSpace;
    }

    public void setSubSpace(final ParamSpace subSpace) {
        this.subSpace = Objects.requireNonNull(subSpace);
    }

}
