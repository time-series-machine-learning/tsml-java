package tsml.classifiers.distance_based.utils.classifier_mixins;

import java.util.logging.Logger;
import org.junit.Assert;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.proximity.RandomSource;
import tsml.classifiers.distance_based.utils.logging.Debugable;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.logging.Loggable;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import weka.core.Debug.Random;
import weka.core.Instances;

/**
 * Purpose: base classifier implementing all common interfaces. Note, this is only for implementation ubiquitous to
 * *every single classifier*. Don't add any optional / unused interface implementation, that should be done via mixins
 * in your concrete class.
 * <p>
 * Contributors: goastler
 */
public abstract class BaseClassifier extends EnhancedAbstractClassifier implements Rebuildable, ParamHandler, Copy,
    Debugable, TrainEstimateable,
    Loggable, RandomSource {

    private Logger logger = LogUtils.buildLogger(this);
    private boolean built = false;
    private boolean rebuild = true;
    private boolean debug = false;
    private boolean seedSet = false;

    public BaseClassifier() {

    }

    public BaseClassifier(boolean a) {
        super(a);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        if(isRebuild()) {
            Assert.assertNotNull(trainData);
            // check the seed has been set
            if(!seedSet) {
                throw new IllegalStateException("seed not set");
            }
            // we're rebuilding so set the seed / params, etc, using super
            super.buildClassifier(trainData);
            setRebuild(false);
        }
        // assume that child classes will set built to true if/when they're built. We'll set it to false here in case
        // it's already been set to true from a previous call
        setBuilt(false);
    }

    @Override
    public void setClassifierName(String classifierName) {
        Assert.assertNotNull(classifierName);
        super.setClassifierName(classifierName);
        logger = LogUtils.buildLogger(classifierName);
    }

    @Override
    public boolean isDebug() {
        return debug;
    }

    @Override
    public void setDebug(boolean debug) {
        this.debug = debug;
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

    public boolean isBuilt() {
        return built;
    }

    protected void setBuilt(boolean built) {
        this.built = built;
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
}
