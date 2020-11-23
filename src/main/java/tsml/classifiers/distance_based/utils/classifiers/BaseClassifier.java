package tsml.classifiers.distance_based.utils.classifiers;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.OutOfBagEvaluator;
import evaluation.storage.ClassifierResults;

import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Assert;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: base classifier implementing all common interfaces. Note, this is only for implementation ubiquitous to
 * *every single classifier*. Don't add any optional / unused interface implementation, that should be done via mixins
 * in your concrete class.
 * <p>
 * Contributors: goastler
 */
public abstract class BaseClassifier extends EnhancedAbstractClassifier implements Rebuildable, ParamHandler, Copier, TrainEstimateable, Loggable {
    // method of logging
    private transient Logger log = LogUtils.DEFAULT_LOG;
    // whether the classifier is to be built from scratch or not. Set this to true to incrementally improve the model on every buildClassifier call
    private boolean rebuild = true;
    // whether the seed has been set
    private boolean seedSet = false;
    // whether to (re)generate train estimate. Useful with rebuild to incrementally improve classifier.
    private boolean rebuildTrainEstimateResults = true;

    protected BaseClassifier() {
        this(false);
    }

    protected BaseClassifier(boolean a) {
        super(a);
    }

    @Override public void setDebug(final boolean b) {
        super.setDebug(b);
        if(debug) {
            setLogLevel(Level.ALL);
        } else {
            log = LogUtils.DEFAULT_LOG;
        }
    }
    
    protected Logger getLog() {
        return log;
    }

    @Override public void setClassifierName(final String classifierName) {
        super.setClassifierName(Objects.requireNonNull(classifierName));
        // set the log level to the current level. If this is different to the default logger then a new logger will be made with the new classifier name
        setLogLevel(getLogLevel());
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        if(rebuild) {
            // reset train results
            trainResults = new ClassifierResults();
            // check the seed has been set
            if(!seedSet) {
                throw new IllegalStateException("seed not set");
            }
            // we're rebuilding so set the seed / params, etc, using super
            super.buildClassifier(Objects.requireNonNull(trainData));
        }
    }

    @Override public Level getLogLevel() {
        return log.getLevel();
    }

    @Override public void setLogLevel(final Level level) {
        log = LogUtils.updateLogLevel(this, log, Objects.requireNonNull(level));
    }

    @Override
    public ParamSet getParams() {
        return new ParamSet();
    }

    @Override
    public void setParams(ParamSet params) throws Exception {
        Objects.requireNonNull(params);
    }

    @Override
    public String getParameters() {
        return super.getParameters() + ",params," + getParams().toString();
    }

    public boolean isRebuild() {
        return rebuild;
    }

    public void setRebuild(boolean rebuild) {
        this.rebuild = rebuild;
    }

    @Override
    public void setSeed(int seed) {
        super.setSeed(seed);
        seedSet = true;
    }

    @Override
    public abstract double[] distributionForInstance(final Instance instance) throws Exception;

}
