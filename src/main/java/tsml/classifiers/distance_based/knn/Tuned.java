package tsml.classifiers.distance_based.knn;

import evaluation.storage.ClassifierResults;
import org.junit.Assert;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.Configurer;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.stats.scoring.ResultsScorer;
import weka.core.Instance;
import weka.core.Instances;

public class Tuned extends BaseClassifier {

    private ParamSpaceBuilder paramSpaceBuilder;
    private ParamSpace paramSpace;

    private ResultsScorer resultsScorer = ClassifierResults::getAcc;
    private TuningAgent tuningAgent;

    public enum Config implements Configurer<Tuned> {
        DEFAULT() {
            @Override public Tuned configure(final Tuned classifier) {
                return classifier;
            }
        },
        ;
    }

    public Tuned() {
        Config.DEFAULT.configure(this);
    }

    private boolean shouldExplore() {
        return true;
    }

    private ParamSet pickParameter() {
        return null;
    }

    private EnhancedAbstractClassifier pickClassifier() {
        return null;
    }

    private void improveClassifier(EnhancedAbstractClassifier classifier) {

    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        super.buildClassifier(trainData);
        Assert.assertFalse(paramSpaceBuilder == null && paramSpace == null);
        if(isRebuild()) {
            // build the param space if set
            if(paramSpaceBuilder != null) {
                paramSpace = paramSpaceBuilder.build(trainData);
            }
            tuningAgent.buildAgent(trainData);
        }
        while(tuningAgent.hasNext()) {
            final Benchmark benchmark = tuningAgent.next();
            final EnhancedAbstractClassifier classifier = benchmark.getClassifier();
            classifier.setEstimateOwnPerformance(true);
            classifier.buildClassifier(trainData);
            final ClassifierResults results = classifier.getTrainResults();
            final double score = resultsScorer.score(results);
            benchmark.setScore(score);
            tuningAgent.feedback(benchmark);
        }
    }

    @Override public double[] distributionForInstance(final Instance instance) throws Exception {
        throw new UnsupportedOperationException();
    }

    public ParamSpace getParamSpace() {
        return paramSpace;
    }

    public void setParamSpace(final ParamSpace paramSpace) {
        this.paramSpace = paramSpace;
    }

    public ParamSpaceBuilder getParamSpaceBuilder() {
        return paramSpaceBuilder;
    }

    public void setParamSpaceBuilder(final ParamSpaceBuilder paramSpaceBuilder) {
        this.paramSpaceBuilder = paramSpaceBuilder;
    }
}
