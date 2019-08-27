package timeseriesweka.classifiers.distance_based.ee;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.distance_based.ee.selection.KBestSelector;
import timeseriesweka.classifiers.distance_based.ee.selection.Selector;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.util.Iterator;

public class Tuned implements Iterator<Benchmark> {
    private Iterator<AbstractClassifier> iterator;
    private KBestSelector<Benchmark, Double> selector;
    private Instances trainInstances;
    private Evaluator evaluator;

    public KBestSelector<Benchmark, Double> getSelector() {
        return selector;
    }

    public void setSelector(KBestSelector<Benchmark, Double> selector) {
        this.selector = selector;
    }

    public Iterator<AbstractClassifier> getIterator() {
        return iterator;
    }

    public void setIterator(Iterator<AbstractClassifier> iterator) {
        this.iterator = iterator;
    }


    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public Benchmark next() {
        AbstractClassifier classifier = iterator.next();
        ClassifierResults results = null;
        try {
            results = evaluator.evaluate(classifier, trainInstances);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        Benchmark benchmark = new Benchmark(classifier, results);
        selector.add(benchmark);
        return benchmark;
    }

    public Instances getTrainInstances() {
        return trainInstances;
    }

    public void setTrainInstances(Instances trainInstances) {
        this.trainInstances = trainInstances;
    }

    public Evaluator getEvaluator() {
        return evaluator;
    }

    public void setEvaluator(Evaluator evaluator) {
        this.evaluator = evaluator;
    }
}
