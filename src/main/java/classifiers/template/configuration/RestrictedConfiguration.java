package classifiers.template.configuration;

import weka.core.OptionHandler;

import java.util.Enumeration;

public class RestrictedConfiguration implements OptionHandler {



    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {

    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
