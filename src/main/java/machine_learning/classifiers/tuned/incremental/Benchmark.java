package machine_learning.classifiers.tuned.incremental;

import evaluation.storage.ClassifierResults;
import utilities.StrUtils;
import weka.classifiers.Classifier;

import java.io.Serializable;
import java.util.function.Function;

public class Benchmark implements Serializable, Comparable<Benchmark> {
    private ClassifierResults results;
    private Classifier classifier;
    private final int id;
    private double score = -1;

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    public double score(Function<ClassifierResults, Double> scorer) {
        setScore(scorer.apply(results));
        return getScore();
    }

    public Benchmark(Benchmark benchmark) {
        this(benchmark.classifier, benchmark.results, benchmark.id);
    }

    public Benchmark(int id) {
        this.id = id;
    }

    public Benchmark(Classifier classifier, ClassifierResults results, int id) {
        this.id = id;
        setClassifier(classifier);
        setResults(results);
    }

    public ClassifierResults getResults() {
        return results;
    }

    public void setResults(ClassifierResults results) {
        this.results = results;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    @Override
    public int hashCode() {
        return id;
    }

    @Override
    public boolean equals(Object o) {
        if(o == this) {
            return true;
        }
        if(!(o instanceof Benchmark)) {
            return false;
        }
        return o.hashCode() == hashCode();
    }

    @Override public String toString() {
        return "Benchmark{" +
            "id=" + id +
            ", results=" + results.getAcc() +
            ", classifier=" + StrUtils.toOptionValue(classifier) +
            '}';
    }

    @Override
    public int compareTo(Benchmark benchmark) {
        return Double.compare(score, benchmark.score);
    }
}
