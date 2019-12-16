package utilities;

import evaluation.tuning.ParameterSet;
import weka.core.OptionHandler;
import weka.core.UnsupportedAttributeTypeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;

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
}
