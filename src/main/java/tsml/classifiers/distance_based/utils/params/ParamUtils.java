package tsml.classifiers.distance_based.utils.params;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ParamUtils {

    private ParamUtils() {

    }

    public static <A> void setMultiValuedParam(ParamSet paramSet, String name, Consumer<List<A>> setter,
        Function<String, Object> fromStringFunction) {
        // get the values for the parameter name
        final List<Object> values = paramSet.get(name);
        // if there's no associated values then bail
        if(values == null) {
            return;
        }
        // get the single parameter value
        List<A> castValues = new ArrayList<>();
        for(int i = 0; i < values.size(); i++) {
            Object value = values.get(i);
            if(value instanceof String) {
                value = fromStringFunction.apply((String) value);
            }
            // cast value to class and set param
            A castValue;
            try {
                castValue = (A) value;
            } catch(ClassCastException e) {
                IllegalArgumentException ex = new IllegalArgumentException(
                    "cannot cast {" + value + "}");
                ex.addSuppressed(e);
                throw ex;
            }
            castValues.add(castValue);
        }
        setter.accept(castValues);
    }

    public static <A> void setSingleValuedParam(ParamSet paramSet, String name, Consumer<A> setter,
        Function<String, Object> fromStringFunction) {
        // get the values for the parameter name
        final List<Object> values = paramSet.get(name);
        // if there's no associated values then bail
        if(values == null) {
            return;
        }
        // if there isn't only one of these values then throw exception - this isn't a multi-valued parameter
        if(values.size() != 1) {
            throw new IllegalArgumentException("{" + name + "} cannot accept multiple values {" + values + "}");
        }
        // get the single parameter value
        Object value = values.get(0);
        if(value instanceof String) {
            value = fromStringFunction.apply((String) value);
        }
        // cast value to class and set param
        A castValue;
        try {
            castValue = (A) value;
        } catch(ClassCastException e) {
            IllegalArgumentException ex = new IllegalArgumentException(
                "cannot cast {" + value + "}");
            ex.addSuppressed(e);
            throw ex;
        }
        setter.accept(castValue);
    }
}
