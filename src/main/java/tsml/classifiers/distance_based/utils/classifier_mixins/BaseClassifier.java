package tsml.classifiers.distance_based.utils.classifier_mixins;

import com.google.common.cache.CacheLoader.UnsupportedLoadingOperationException;
import evaluation.storage.ClassifierResults;
import java.lang.reflect.Array;
import java.util.logging.Logger;
import org.junit.Assert;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.proximity.RandomSource;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.logging.Loggable;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: base classifier implementing all common interfaces. Note, this is only for implementation ubiquitous to
 * *every single classifier*. Don't add any optional / unused interface implementation, that should be done via mixins
 * in your concrete class.
 * <p>
 * Contributors: goastler
 */
public abstract class BaseClassifier extends EnhancedAbstractClassifier implements Rebuildable, ParamHandler, Copy, TrainEstimateable,
    Loggable, RandomSource {

    // method of logging
    private Logger logger = LogUtils.buildLogger(this);
    // whether we're initialising the classifier, e.g. setting seed
    private boolean rebuild = true;
    // whether the seed has been set
    private boolean seedSet = false;
    // method of listening for when rebuilding
    private RebuildListener rebuildListener = (final Instances trainData) -> {};
    // has buildClassifier ever been called?
    private boolean firstBuild = true;

    public BaseClassifier() {

    }

    public BaseClassifier(boolean a) {
        super(a);
    }

    public boolean isFirstBuild() {
        return firstBuild;
    }

    protected BaseClassifier setFirstBuild(final boolean firstBuild) {
        this.firstBuild = firstBuild;
        return this;
    }

    public interface RebuildListener {
        void onRebuild(final Instances trainData);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        final boolean firstBuild = isFirstBuild();
        final boolean rebuild = isRebuild();
        final Logger logger = getLogger();
        if(rebuild || firstBuild) {
            logger.info(() -> {
                if(firstBuild) {
                    return "first build";
                } else {
                    return "rebuilding";
                }
            });
            Assert.assertNotNull(trainData);
            // reset train results
            trainResults = new ClassifierResults();
            // check the seed has been set
            if(!seedSet) {
                throw new IllegalStateException("seed not set");
            }
            // we're rebuilding so set the seed / params, etc, using super
            super.buildClassifier(trainData);
            // we're no longer rebuilding
            // assume all subclasses would have save this value before calling this method
            setRebuild(false);
            // this is the first build
            setFirstBuild(false);
            // notify the rebuild listener that we're rebuilding so anything requiring train data knowledge can be
            // setup (this is helpful when tuning over a data dependent range. The range would be setup inside this
            // method and set corresponding variables in this classifier)
            rebuildListener.onRebuild(trainData);
        }
    }

    @Override
    public void setClassifierName(String classifierName) {
        Assert.assertNotNull(classifierName);
        super.setClassifierName(classifierName);
        logger = LogUtils.buildLogger(classifierName);
    }

    public RebuildListener getRebuildListener() {
        return rebuildListener;
    }

    public BaseClassifier setRebuildListener(
        final RebuildListener rebuildListener) {
        this.rebuildListener = rebuildListener;
        return this;
    }

    @Override
    public Logger getLogger() {
        return logger;
    }

    @Override
    public ParamSet getParams() {
        return ParamHandler.super.getParams();
    }

    @Override
    public void setParams(ParamSet params) {
        Assert.assertNotNull(params);
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

    public void setRandom(Random random) {
        Assert.assertNotNull(random);
        rand = random;
    }

    @Override
    public double[] distributionForInstance(final Instance instance) throws Exception {
        throw new UnsupportedOperationException();
    }

    @Override
    public double classifyInstance(final Instance instance) throws Exception {
        double[] distribution = distributionForInstance(instance);
        return ArrayUtilities.argMax(distribution);
    }
}
