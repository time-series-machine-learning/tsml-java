package classifiers.tuning;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import classifiers.distance_based.elastic_ensemble.iteration.ParameterSetIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RandomIterator;
import classifiers.template.config.TemplateConfig;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.util.Comparator;
import java.util.Iterator;
import java.util.function.Function;
import java.util.function.Supplier;

public class TunerConfig extends TemplateConfig {

    private Function<Instances, AbstractIterator<AbstractClassifier>> classifierIteratorGetter;
    private AbstractIterator<AbstractClassifier> classifierIterator;
    private Comparator<ClassifierResults> comparator = Comparator.comparingDouble(ClassifierResults::getAcc);
    // todo keep more than 1, perhaps they should then be ensembled?

    public TunerConfig(Function<Instances, AbstractIterator<AbstractClassifier>> classifierIteratorGetter) {
        this.classifierIteratorGetter = classifierIteratorGetter;
    }

    public TunerConfig(AbstractIterator<AbstractClassifier> classifierIterator) {
        this.classifierIterator = classifierIterator;
    }

    public TunerConfig(TunerConfig other) {
        this(other.classifierIteratorGetter);
    }

    @Override
    public TemplateConfig copy() throws Exception {
        throw new UnsupportedOperationException(); // todo
    }

    @Override
    public void copyFrom(Object object) throws Exception {
        throw new UnsupportedOperationException(); // todo
    }

    @Override
    public void setOption(String key, String value) {
        // todo
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }

    public void buildClassifierIterator(Instances trainSet) {
        if(classifierIteratorGetter != null) {
            classifierIterator = classifierIteratorGetter.apply(trainSet);
        }
    }

    public Comparator<ClassifierResults> getComparator() {
        return comparator;
    }

    public void setComparator(Comparator<ClassifierResults> comparator) {
        this.comparator = comparator;
    }

    public AbstractIterator<AbstractClassifier> getClassifierIterator() {
        return classifierIterator;
    }

    public void setClassifierIterator(AbstractIterator<AbstractClassifier> classifierIterator) {
        this.classifierIterator = classifierIterator;
    }
}
