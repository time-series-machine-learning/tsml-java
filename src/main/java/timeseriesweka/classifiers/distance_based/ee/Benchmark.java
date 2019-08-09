package timeseriesweka.classifiers.distance_based.ee;


import evaluation.storage.ClassifierResults;
import weka.classifiers.AbstractClassifier;

public class Benchmark {
    private final AbstractClassifier classifier;
    private final ClassifierResults trainResults;

    public Benchmark(AbstractClassifier classifier, ClassifierResults trainResults) {
        this.classifier = classifier;
        this.trainResults = trainResults;
    }

    public ClassifierResults getTrainResults() {
        return trainResults;
    }

    public AbstractClassifier getClassifier() {
        return classifier;
    }
}

