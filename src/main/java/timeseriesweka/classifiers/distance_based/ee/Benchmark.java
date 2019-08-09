package timeseriesweka.classifiers.distance_based.ee;


import evaluation.storage.ClassifierResults;
import weka.classifiers.AbstractClassifier;

public class Benchmark {
    private final AbstractClassifier classifier;
    private final ClassifierResults results;

    public Benchmark(AbstractClassifier classifier, ClassifierResults results) {
        this.classifier = classifier;
        this.results = results;
    }

    public ClassifierResults getResults() {
        return results;
    }

    public AbstractClassifier getClassifier() {
        return classifier;
    }
}

