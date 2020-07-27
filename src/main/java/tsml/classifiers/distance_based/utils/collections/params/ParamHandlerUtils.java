package tsml.classifiers.distance_based.utils.collections.params;

import tsml.classifiers.distance_based.utils.strings.StrUtils;

import java.util.List;
import java.util.function.Consumer;

public class ParamHandlerUtils {

    public static <A> boolean setParam(ParamSet paramSet, String name, Consumer<A> setter,
                                       Class<A> clazz) throws Exception {
        final List<Object> list = paramSet.get(name);
        if(list == null) {
            return false;
        }
        for(Object value : list) {
            if(value instanceof String) {
                value = StrUtils.fromOptionValue((String) value, clazz);
            }
            setter.accept((A) value);
        }
        return true;
    }
}
