package tsml.classifiers.distance_based.utils.classifiers;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.OutOfBagEvaluator;
import evaluation.storage.ClassifierResults;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Assert;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import utilities.ArrayUtilities;
import utilities.Utilities;
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
    private transient Logger logger = LogUtils.buildLogger(getClass());
    // whether we're initialising the classifier, e.g. setting seed
    private boolean rebuild = true;
    // whether the seed has been set
    private boolean seedSet = false;
    // whether to generate train estimate
    private boolean rebuildTrainEstimateResults = true;

    public BaseClassifier() {
        this(false);
    }

    public BaseClassifier(boolean a) {
        super(a);
    }

    protected Evaluator buildEvaluator() {
        switch(estimator) {
            case OOB:
                final OutOfBagEvaluator outOfBagEvaluator = new OutOfBagEvaluator();
                outOfBagEvaluator.setCloneClassifier(true);
                return outOfBagEvaluator;
            case CV:
                final CrossValidationEvaluator crossValidationEvaluator = new CrossValidationEvaluator();
                crossValidationEvaluator.setCloneClassifiers(true);
                crossValidationEvaluator.setNumFolds(10);
                crossValidationEvaluator.setCloneData(true);
                crossValidationEvaluator.setSetClassMissing(true);
                return crossValidationEvaluator;
            default:
                throw new UnsupportedOperationException("cannot handle " + estimator);
        }
    }

    private void setLogLevelFromDebug() {
        if(logger != null) {
            if(debug) {
                logger.setLevel(Level.FINE);
            } else {
                logger.setLevel(Level.OFF);
            }
        }
    }

    @Override public void setDebug(final boolean b) {
        super.setDebug(b);
        setLogLevelFromDebug();
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        logger.info(() -> {
            String msg = "building " + getClassifierName();
            if(rebuild) {
                msg += " from scratch";
            }
            return msg;
        });
        if(rebuild) {
            // reset rebuild
            // default behaviour is to rebuild initially, then turn off the rebuild flag. Subsequent calls to buildClassifier will result in further building instead of building from scratch every time. If you need to build from scratch, just call setRebuild(true) before calling this method.
            rebuild = false;
            Assert.assertNotNull(trainData);
            // reset train results
            trainResults = new ClassifierResults();
            // check the seed has been set
            if(!seedSet) {
                throw new IllegalStateException("seed not set");
            }
            // we're rebuilding so set the seed / params, etc, using super
            super.buildClassifier(trainData);
        }
    }

    @Override
    public Logger getLogger() {
        return logger;
    }

    @Override public void setLogger(final Logger logger) {
        Assert.assertNotNull(logger);
        this.logger = logger;
    }

    @Override
    public ParamSet getParams() {
        return new ParamSet();
    }

    @Override
    public void setParams(ParamSet params) throws Exception {
        Assert.assertNotNull(params);
    }

    @Override
    public String getParameters() {
        return super.getParameters() + "," + getParams().toString();
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

    public boolean isRebuildTrainEstimateResults() {
        return rebuildTrainEstimateResults;
    }

    public void setRebuildTrainEstimateResults(final boolean rebuildTrainEstimateResults) {
        this.rebuildTrainEstimateResults = rebuildTrainEstimateResults;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] distribution = distributionForInstance(instance);
        return Utilities.argMax(distribution, rand);
    }

}
