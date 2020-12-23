package tsml.classifiers.distance_based.utils.collections.params;

import tsml.classifiers.distance_based.utils.strings.StrUtils;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

public class ParamHandlerUtils {

    public static <A> boolean setParam(ParamSet paramSet, String name, Consumer<A> setter) throws Exception {
        return setParam(paramSet, name, setter, str -> {
            throw new IllegalArgumentException("cannot parse " + str + " to primitive");
        });
    }
    
    /**
     * Set a parameter value specified by the name flag. The parameter value may be parsed from string form into a form accepted by the setter.
     * @param paramSet
     * @param name
     * @param setter
     * @param <A>
     * @return
     * @throws Exception
     */
    public static <A> boolean setParam(ParamSet paramSet, String name, Consumer<A> setter, Function<String, A> parser)
            throws Exception {
        // get the parameter in the paramset under "name"
        final List<Object> list = paramSet.get(name);
        // bail if not available
        if(list == null) {
            return false;
        }
        // for every value associated with the parameter name
        for(Object value : list) {
            // check the value is not in string form
            if(value instanceof String) {
                // if it is then construct the options string into a valid object
                value = StrUtils.fromOptionValue((String) value, parser);
            }
            // and pass to the setter
            try {
                setter.accept((A) value);
            } catch(ClassCastException e) {
                final IllegalStateException ise =
                        new IllegalStateException("cannot cast " + value + " to parameter type");
                ise.addSuppressed(e);
                throw ise;
            }
        }
        return true;
    }
}
