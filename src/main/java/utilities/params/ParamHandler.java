package utilities.params;

import weka.core.OptionHandler;

import java.util.*;
import java.util.function.Consumer;

public interface ParamHandler
    extends OptionHandler {

    @Override
    default String[] getOptions() {
        return getOptionsList().toArray(new String[0]);
    }

    default List<String> getOptionsList() {
        return new ArrayList<>();
    }

    default void setOptionsList(List<String> options) throws
                                                      Exception {

    }

    @Override
    default void setOptions(String[] options) throws
                                      Exception {
        setOptionsList(new ArrayList<>(Arrays.asList(options)));
    }

    @Override
    default Enumeration listOptions() {
        throw new UnsupportedOperationException(); // todo
    }

    default void setParams(ParamSet params) {
//        setOptionsList(params.get);
        // todo
        throw new UnsupportedOperationException();
    }

    default ParamSet getParams() {
        // todo
        throw new UnsupportedOperationException();
    }

    static <A> void setParam(ParamSet params, String name, Consumer<A> setter, Class<? extends A> clazz) {
        List<ParamSet.ParamValue> paramSets = params.get(name);
        for(ParamSet.ParamValue paramSet : paramSets) {
            Object value = paramSet.getValue();
            if(value instanceof ParamHandler) {
                for(ParamSet subParamSet : paramSet.getParamSetList()) {
                    ((ParamHandler) value).setParams(subParamSet);
                }
            }
            setter.accept((clazz.cast(value)));
        }
    }
}
