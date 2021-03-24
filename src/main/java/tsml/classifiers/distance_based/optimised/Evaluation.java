package tsml.classifiers.distance_based.optimised;

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.TSClassifier;
import weka.classifiers.Classifier;

import java.util.Objects;

public class Evaluation implements Comparable<Evaluation> {

    public ClassifierResults getResults() {
        return results;
    }

    public void setResults(final ClassifierResults results) {
        this.results = results;
        if(results != null) {
            score = scorer.score(results);
        }
    }
    
    public Evaluation(final int id) {
        this.id = id;
        setScorer(ClassifierResults::getAcc);
    }

    private TSClassifier classifier;
    private Evaluator evaluator;
    private ClassifierResults results;
    private final int id;
    private double score;
    private ResultsScorer scorer;
    private boolean explore = true;
    
    public boolean isExploit() {
        return !isExplore();
    }
    
    public void setExploit() {
        explore = false;
    }
    
    public TSClassifier getClassifier() {
        return classifier;
    }
    
    public int getId() {
        return id;
    }

    @Override public boolean equals(final Object o) {
        if(!(o instanceof Evaluation)) {
            return false;
        }
        final Evaluation that = (Evaluation) o;
        return that.id == id;
    }

    @Override public int hashCode() {
        return id;
    }

    public Evaluator getEvaluator() {
        return evaluator;
    }

    public void setClassifier(final TSClassifier classifier) {
        this.classifier = Objects.requireNonNull(classifier);
    }

    public void setEvaluator(final Evaluator evaluator) {
        this.evaluator = Objects.requireNonNull(evaluator);
    }
    
    public void setClassifier(Classifier classifier) {
        setClassifier(TSClassifier.wrapClassifier(classifier));
    }

    @Override public String toString() {
        return String.valueOf(id);
    }
    
    public String toStringVerbose() {
        return (explore ? "explore" : "exploit") + " " + id + " ( " + classifier.toString() + " )" + " : " + score;
    }

    public ResultsScorer getScorer() {
        return scorer;
    }

    public void setScorer(final ResultsScorer scorer) {
        this.scorer = scorer;
    }

    public double getScore() {
        return score;
    }

    public boolean isExplore() {
        return explore;
    }

    public void setExplore() {
        this.explore = explore;
    }

    @Override public int compareTo(final Evaluation evaluation) {
        return Double.compare(score, evaluation.getScore());
    }
}
