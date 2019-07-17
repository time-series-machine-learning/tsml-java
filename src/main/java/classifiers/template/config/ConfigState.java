package classifiers.template.config;

import java.util.Comparator;
import java.util.function.Supplier;

public class ConfigState<A extends TemplateConfig> {
    private A previous;
    private A current;
    private A next;
    private final Supplier<A> supplier;
    private final Comparator<A> trainComparator;
    private final Comparator<A> testComparator;

    public ConfigState(Supplier<A> supplier,
                       final Comparator<A> trainComparator,
                       final Comparator<A> testComparator) {
        this.supplier = supplier;
        previous = supplier.get();
        current = supplier.get();
        next = supplier.get();
        this.trainComparator = trainComparator;
        this.testComparator = testComparator;
    }

    public A getNext() {
        return next;
    }

    public A getCurrent() {
        return current;
    }

    public A getPrevious() {
        return previous;
    }

    public boolean mustResetTrain() {
        return trainComparator.compare(previous, current) > 0;
    }

    public boolean mustResetTest() {
        return testComparator.compare(previous, current) > 0;
    }

    public void shift() throws
                        Exception {
        previous = current;
        current = next;
        next = supplier.get();
        next.copyFrom(current);
    }
}
