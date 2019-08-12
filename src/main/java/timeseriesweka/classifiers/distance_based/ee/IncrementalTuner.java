package timeseriesweka.classifiers.distance_based.ee;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.distance_based.ee.selection.KBestSelector;
import utilities.iteration.AbstractIterator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.util.Iterator;

public class IncrementalTuner
    implements Iterator<Benchmark> {
    private Iterator<AbstractClassifier> iterator;
    private KBestSelector<Benchmark, Double> selector;
    private Evaluator evaluator;
    private Instances instances;

    public void setIterator(final Iterator<AbstractClassifier> iterator) {
        this.iterator = iterator;
    }

    public void setSelector(final KBestSelector<Benchmark, Double> selector) {
        this.selector = selector;
    }

    public Evaluator getEvaluator() {
        return evaluator;
    }

    public void setEvaluator(final Evaluator evaluator) {
        this.evaluator = evaluator;
    }

    public Instances getInstances() {
        return instances;
    }

    public void setInstances(final Instances instances) {
        this.instances = instances;
    }

    public KBestSelector<Benchmark, Double> getSelector() {
        return selector;
    }

    public Iterator<AbstractClassifier> getIterator() {
        return iterator;
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public Benchmark next() {
        AbstractClassifier classifier = iterator.next();
        ClassifierResults results;
        try {
            results = evaluator.evaluate(classifier, instances);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        Benchmark benchmark = new Benchmark(classifier, results);
        selector.add(benchmark);
        return benchmark;
    }

}
