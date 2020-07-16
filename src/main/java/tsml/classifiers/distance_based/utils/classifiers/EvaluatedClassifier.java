package tsml.classifiers.distance_based.utils.classifiers;

import evaluation.evaluators.Evaluator;
import org.junit.Assert;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.system.random.RandomSource;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import java.util.Random;
import java.util.logging.Logger;

public class EvaluatedClassifier extends BaseClassifier {

    private Classifier classifier;
    private Evaluator evaluator;
    private boolean buildClassifierAfterEvaluation = true;

    public EvaluatedClassifier() {}

    public EvaluatedClassifier(Classifier classifier, Evaluator evaluator) {
        setClassifier(classifier);
        setEvaluator(evaluator);
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        Assert.assertNotNull(classifier);
        Assert.assertNotNull(evaluator);
        super.buildClassifier(trainData);
        trainResults = evaluator.evaluate(classifier, trainData);
        ResultUtils.setInfo(trainResults, classifier, trainData);
        if(buildClassifierAfterEvaluation) {
            classifier.buildClassifier(trainData);
        }
    }

    @Override public double[] distributionForInstance(final Instance instance) throws Exception {
        return classifier.distributionForInstance(instance);
    }

    @Override public double classifyInstance(final Instance instance) throws Exception {
        return classifier.classifyInstance(instance);
    }

    @Override public void setLogger(final Logger logger) {
        super.setLogger(logger);
        if(classifier instanceof Loggable) {
            ((Loggable) classifier).setLogger(logger);
        }
    }

    @Override public void setDebug(final boolean b) {
        super.setDebug(b);
        if(classifier instanceof EnhancedAbstractClassifier) {
            ((EnhancedAbstractClassifier) classifier).setDebug(b);
        }
    }

    @Override public ParamSet getParams() {
        // todo add params for this class
        if(classifier instanceof ParamHandler) {
            return ((ParamHandler) classifier).getParams();
        } else {
            return super.getParams();
        }
    }

    @Override public void setParams(final ParamSet params) throws Exception {
        // todo add params for this class
        super.setParams(params);
        if(classifier instanceof ParamHandler) {
            ((ParamHandler) classifier).setParams(params);
        }
    }

    @Override public void setRebuild(final boolean rebuild) {
        super.setRebuild(rebuild);
        if(classifier instanceof Rebuildable) {
            ((Rebuildable) classifier).setRebuild(rebuild);
        }
    }

    @Override public void setSeed(final int seed) {
        super.setSeed(seed);
        if(classifier instanceof Randomizable) {
            ((Randomizable) classifier).setSeed(seed);
        }
    }

    @Override public void setRandom(final Random random) {
        super.setRandom(random);
        if(classifier instanceof RandomSource) {
            ((RandomSource) classifier).setRandom(random);
        }
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(final Classifier classifier) {
        Assert.assertNotNull(classifier);
        this.classifier = classifier;
    }

    public Evaluator getEvaluator() {
        return evaluator;
    }

    public void setEvaluator(final Evaluator evaluator) {
        Assert.assertNotNull(evaluator);
        this.evaluator = evaluator;
    }

    public boolean isBuildClassifierAfterEvaluation() {
        return buildClassifierAfterEvaluation;
    }

    public void setBuildClassifierAfterEvaluation(final boolean buildClassifierAfterEvaluation) {
        this.buildClassifierAfterEvaluation = buildClassifierAfterEvaluation;
    }
}
