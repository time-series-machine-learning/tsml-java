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
        return getParams().getOptionsList();
    }

    default void setOptionsList(List<String> options) throws
                                                      Exception {
        ParamSet params = new ParamSet();
        params.setOptionsList(options);
        setParams(params);
    }

    @Override
    default void setOptions(String[] options) throws
                                      Exception {
        setOptionsList(new ArrayList<>(Arrays.asList(options))); // todo replace with view
    }

    @Override
    default Enumeration listOptions() {
        return Collections.enumeration(listParams());
    }

    default void setParams(ParamSet param) {
        throw new UnsupportedOperationException("param setting not supported");
    }

    default ParamSet getParams() {
        return new ParamSet();
    }

    // todo some kind of check against listparams to ensure it's settable
    static <A> void setParam(ParamSet params, String name, Consumer<A> setter, Class<? extends A> clazz) {
        List<ParamSet.ParamValue> paramSets = params.get(name);
        if(paramSets == null) {
            return;
        }
        for(ParamSet.ParamValue paramSet : paramSets) {
            Object value = paramSet.getValue();
            if(value instanceof ParamHandler) {
                ParamHandler paramHandler = (ParamHandler) value;
                for(ParamSet subParam : paramSet.getParamList()) {
                    paramHandler.setParams(subParam);
                }
            }
            setter.accept((clazz.cast(value)));
        }
    }

    default Set<String> listParams() {
        // todo use getParams to populate this
        throw new UnsupportedOperationException("param list not specified");
    }
}
