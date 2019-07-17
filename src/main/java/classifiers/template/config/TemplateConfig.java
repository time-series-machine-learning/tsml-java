package classifiers.template.config;

import utilities.Copyable;
import utilities.IndividualOptionHandler;

public abstract class TemplateConfig
        implements Copyable, IndividualOptionHandler {

    public TemplateConfig() {}

    public TemplateConfig(Object other) throws
                                          Exception {
        copyFrom(other);
    }

    @Override
    public abstract TemplateConfig copy() throws
                                          Exception;


}
