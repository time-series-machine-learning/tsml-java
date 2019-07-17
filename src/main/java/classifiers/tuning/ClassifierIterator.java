package classifiers.tuning;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import evaluation.tuning.ParameterSet;
import weka.classifiers.AbstractClassifier;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

public class ClassifierIterator extends AbstractIterator<AbstractClassifier> {

    private AbstractIterator<ParameterSet> iterator;
    private Supplier<AbstractClassifier> supplier;
    private final List<AbstractClassifier> added = new ArrayList<>();

    public AbstractIterator<ParameterSet> getIterator() {
        return iterator;
    }

    public void setIterator(final AbstractIterator<ParameterSet> iterator) {
        this.iterator = iterator;
    }

    public Supplier<AbstractClassifier> getSupplier() {
        return supplier;
    }

    public void setSupplier(final Supplier<AbstractClassifier> supplier) {
        this.supplier = supplier;
    }

    public ClassifierIterator() {}

    public ClassifierIterator(Supplier<AbstractClassifier> supplier, AbstractIterator<ParameterSet> iterator) {
        this.supplier = supplier;
        this.iterator = iterator;
    }

    @Override
    public ClassifierIterator iterator() {
        throw new UnsupportedOperationException(); // todo
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public AbstractClassifier next() {
        AbstractClassifier classifier;
        if(added.isEmpty()) {
            classifier = supplier.get();
        } else {
            classifier = added.remove(0);
        }
        ParameterSet parameterSet = iterator.next();
        try {
            classifier.setOptions(parameterSet.getOptions());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return classifier;
    }

    @Override
    public void remove() {
        iterator.remove();
    }

    @Override
    public void add(AbstractClassifier classifier) {
        added.add(classifier);
    }
}
