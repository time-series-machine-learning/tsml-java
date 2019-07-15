package classifiers.template.configuration;

import utilities.Copyable;
import utilities.IndividualOptionHandler;

public abstract class TemplateConfig<A extends TemplateConfig<A>>
    extends IndividualOptionHandler
    implements Copyable<TemplateConfig<A>> {

    public TemplateConfig() {}

    public TemplateConfig(A other) throws
                                          Exception {
        copyFrom(other);
    }

    public boolean mustResetTrain(final A other) {
        return false;
    }

    public boolean mustResetTest(final A other) {
        return false;
    }

}
