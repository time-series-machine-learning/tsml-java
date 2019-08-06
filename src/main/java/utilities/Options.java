package utilities;

import weka.core.OptionHandler;

public interface Options extends OptionHandler {
    void setOption(String key, String value);

    default void setOptions(String[] options) throws
                                              Exception {
        StringUtilities.forEachPair(options, this::setOption);
    }
}
