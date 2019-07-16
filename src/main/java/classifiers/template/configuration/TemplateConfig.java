package classifiers.template.configuration;

import utilities.Copyable;
import utilities.IndividualOptionHandler;

public abstract class TemplateConfig
    extends IndividualOptionHandler
    implements Copyable {

    public TemplateConfig() {}

    public TemplateConfig(Object other) throws
                                          Exception {
        copyFrom(other);
    }

    @Override
    public abstract TemplateConfig copy() throws
                                          Exception;


}
