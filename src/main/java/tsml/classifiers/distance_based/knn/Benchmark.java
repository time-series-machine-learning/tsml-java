package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.EnhancedAbstractClassifier;
import weka.classifiers.Classifier;

import java.util.Objects;

public class Benchmark {
    private double score = -1;
    private EnhancedAbstractClassifier classifier;
    private static int idCounter = 0;
    private final int id = idCounter++;

    public Benchmark(EnhancedAbstractClassifier classifier) {
        setClassifier(classifier);
    }

    public EnhancedAbstractClassifier getClassifier() {
        return classifier;
    }

    public void setClassifier(final EnhancedAbstractClassifier classifier) {
        this.classifier = classifier;
    }

    public double getScore() {
        return score;
    }

    public void setScore(final double score) {
        this.score = score;
    }

    @Override public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(!(o instanceof Benchmark)) {
            return false;
        }
        final Benchmark benchmark = (Benchmark) o;
        return id == benchmark.id;
    }

    @Override public int hashCode() {
        return id;
    }
}
