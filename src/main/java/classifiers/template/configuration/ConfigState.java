package classifiers.template.configuration;

import java.util.function.Supplier;

public class ConfigState<A extends TemplateConfig<A>> {
    private A previous;
    private A current;
    private A next;
    private final Supplier<A> supplier;

    public ConfigState(Supplier<A> supplier) {
        this.supplier = supplier;
        previous = supplier.get();
        current = supplier.get();
        next = supplier.get();
    }

    public A getNextConfig() {
        return next;
    }

    public A getCurrentConfig() {
        return current;
    }

    public A getPreviousConfig() {
        return previous;
    }

    public boolean mustResetTrain() {
        return previous.mustResetTrain(current);
    }

    public boolean mustResetTest() {
        return previous.mustResetTest(current);
    }

    public void shift() {
        previous = current;
        current = next;
        next = supplier.get();
    }
}
