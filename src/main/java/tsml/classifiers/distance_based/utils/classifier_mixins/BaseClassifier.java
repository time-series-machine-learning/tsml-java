package tsml.classifiers.distance_based.utils.classifier_mixins;

import java.util.Objects;
import java.util.logging.Logger;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.logging.Debugable;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.logging.Loggable;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import weka.core.Instances;

/**
 * Purpose: base classifier implementing all common interfaces. Note, this is only for implementation ubiquitous to
 * *every single classifier*. Don't add any optional / unused interface implementation, that should be done via
 * mixins in your concrete class.
 * <p>
 * Contributors: goastler
 */
public abstract class BaseClassifier extends EnhancedAbstractClassifier implements Rebuildable, ParamHandler, Copy,
    Debugable,
    Loggable {

    private Logger logger = LogUtils.buildLogger(this);
    private boolean built = false;
    private boolean rebuild = true;
    private boolean debug = false;

    public BaseClassifier() {

    }

    public BaseClassifier(boolean a) {
        super(a);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        if(isRebuild()) {
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
        super.setClassifierName(classifierName);
        if(classifierName != null) {
            logger = LogUtils.buildLogger(classifierName);
        } else {
            logger = LogUtils.buildLogger(this);
        }
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
}
