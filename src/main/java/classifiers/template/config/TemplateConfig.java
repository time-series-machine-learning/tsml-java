package classifiers.template.config;

import utilities.Copyable;
import weka.core.OptionHandler;

import java.util.Enumeration;

public abstract class TemplateConfig
        implements Copyable,
                   OptionHandler {

    public TemplateConfig() {}

    public TemplateConfig(Object other) throws
                                          Exception {
        copyFrom(other);
    }

    @Override
    public abstract TemplateConfig copy() throws
                                          Exception;

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }
}
