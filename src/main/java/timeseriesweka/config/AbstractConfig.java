package timeseriesweka.config;

import utilities.Copyable;
import utilities.Options;
import weka.core.OptionHandler;

import java.util.Enumeration;

public abstract class AbstractConfig
        implements Copyable,
                   Options {

    public AbstractConfig() {}

    public AbstractConfig(Object other) throws
                                          Exception {
        copyFrom(other);
    }

    @Override
    public abstract AbstractConfig copy() throws
                                          Exception;


    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }
}
