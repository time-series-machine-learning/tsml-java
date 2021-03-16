package tsml.classifiers.distance_based.utils.classifiers;

import evaluation.storage.ClassifierResults;

import java.util.Objects;
import java.util.Random;
import java.util.logging.Logger;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.utils.system.copy.Copier;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.system.random.Randomised;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Purpose: base classifier implementing all common interfaces. Note, this is only for implementation ubiquitous to
 * *every single classifier*. Don't add any optional / unused interface implementation, that should be done via mixins
 * in your concrete class.
 * <p>
 * Contributors: goastler
 */
public abstract class BaseClassifier extends EnhancedAbstractClassifier implements Rebuildable, ParamHandler, Copier, TrainEstimateable, Loggable,
                                                                                           Randomised {
    
    private transient Logger log = LogUtils.getLogger(getClass());
    
    // whether the classifier is to be built from scratch or not. Set this to true to incrementally improve the model on every buildClassifier call
    private boolean rebuild = true;

    protected BaseClassifier() {
        this(false);
    }

    protected BaseClassifier(boolean a) {
        super(a);
    }

    @Override public Logger getLogger() {
        return log;
    }

    @Override public void setLogger(final Logger logger) {
        log = Objects.requireNonNull(logger);
    }

    @Override public void setClassifierName(final String classifierName) {
        super.setClassifierName(Objects.requireNonNull(classifierName));
    }

    @Override public void buildClassifier(final TimeSeriesInstances trainData) throws Exception {
        if(rebuild) {
            // reset train results
            trainResults = new ClassifierResults();
            // check the seed has been set
            checkRandom();
            // we're rebuilding so set the seed / params, etc, using super
            super.buildClassifier(Converter.toArff(Objects.requireNonNull(trainData)));
        }
    }

    @Override
    public final void buildClassifier(Instances trainData) throws Exception {
        buildClassifier(Converter.fromArff(trainData));
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
        setRandom(new Random(seed));
    }

    @Override
    public final double[] distributionForInstance(final Instance instance) throws Exception {
        return distributionForInstance(Converter.fromArff(instance));
    }

    @Override public abstract double[] distributionForInstance(final TimeSeriesInstance inst) throws Exception;

    @Override public String[] getOptions() {
        return ParamHandler.super.getOptions();
    }

    @Override public void setOptions(final String[] options) throws Exception {
        ParamHandler.super.setOptions(options);
    }

    @Override public String toString() {
        final String options = Utils.joinOptions(getOptions());
        if(!options.isEmpty()) {
            return getClassifierName() + " " + options;
        } else {
            return getClassifierName();
        }
    }
}
