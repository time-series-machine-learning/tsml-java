package tsml.classifiers.distance_based.utils.collections.params;

import tsml.classifiers.distance_based.utils.strings.StrUtils;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

public class ParamHandlerUtils {

    /**
     * Set a non-primitive parameter. This assumes that the corresponding value in string form. The string must be in the form of "{className} {key-value pairs}". This is handled internally in ParamSet, however. The key gotcha here is this method cannot handle primitives!!!
     * @param paramSet
     * @param name
     * @param setter
     * @param <A>
     * @return
     */
    public static <A> boolean setParam(ParamSet paramSet, String name, Consumer<A> setter) throws Exception {
        return setParam(paramSet, name, setter, s -> {
            throw new UnsupportedOperationException("parsing non-primitive type not supported. When setting parameters you must provide a function to parse primitive values otherwise they are interpretted as objects and instantiated by class name. This error occurred whilst trying to instantiate a class called: " + s);
        });
    }

    /**
     * Set a primitive or simply parameter which can be parsed from string form. E.g. the parameter value from the paramSet may be "5.5". The consumer accepts doubles, so the parser must use something similar to Double.parseDouble to convert the string to a double ready to pass to the consumer / setter.
     * @param paramSet
     * @param name
     * @param setter
     * @param parser
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
