package utilities.iteration;

import evaluation.tuning.ParameterSet;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;

import java.util.function.Supplier;

public class ClassifierIterator
    extends AbstractIterator<AbstractClassifier> {

    private AbstractIterator<ParameterSet> parameterSetIterator;
    private Supplier<AbstractClassifier> supplier;

    public ClassifierIterator() {

    }

    public ClassifierIterator(Supplier<AbstractClassifier> supplier, AbstractIterator<ParameterSet> parameterSetIterator) {
        setSupplier(supplier);
        setParameterSetIterator(parameterSetIterator);
    }

    @Override
    public AbstractIterator<AbstractClassifier> iterator() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean hasNext() {
        return parameterSetIterator.hasNext();
    }

    @Override
    public AbstractClassifier next() {
        AbstractClassifier classifier = supplier.get();
        ParameterSet parameterSet = parameterSetIterator.next();
        try {
            classifier.setOptions(parameterSet.getOptions());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        return classifier;
    }

    @Override
    public void remove() {
        parameterSetIterator.remove();
    }

    @Override
    public void add(final AbstractClassifier classifier) {
        try {
            parameterSetIterator.add(new ParameterSet(classifier.getOptions()));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    public AbstractIterator<ParameterSet> getParameterSetIterator() {
        return parameterSetIterator;
    }

    public void setParameterSetIterator(final AbstractIterator<ParameterSet> parameterSetIterator) {
        this.parameterSetIterator = parameterSetIterator;
    }

    public Supplier<AbstractClassifier> getSupplier() {
        return supplier;
    }

    public void setSupplier(final Supplier<AbstractClassifier> supplier) {
        this.supplier = supplier;
    }
}
