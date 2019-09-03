package timeseriesweka.classifiers.distance_based.ee;

import evaluation.storage.ClassifierResults;
import timeseriesweka.classifiers.ParameterSplittable;
import timeseriesweka.classifiers.SeedableClassifier;
import timeseriesweka.classifiers.TrainAccuracyEstimator;
import utilities.ArrayUtilities;
import utilities.StringUtilities;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.logging.Logger;

public class OptimisedClassifier extends AbstractClassifier implements SeedableClassifier, TrainAccuracyEstimator, ParameterSplittable {

    private Tuned tuned;
    private Logger logger = Logger.getLogger(OptimisedClassifier.class.getCanonicalName());
    private Benchmark best;
    private Random trainRandom = new Random();
    private Long trainSeed;
    private Long testSeed;
    private Function<Instances, Tuned> tunedFunction;
    private int index = -1;

    public Function<Instances, Tuned> getTunedFunction() {
        return tunedFunction;
    }

    public void setTunedFunction(Function<Instances, Tuned> tunedFunction) {
        this.tunedFunction = tunedFunction;
    }

    private void setupTrain(Instances trainInstances) {
        best = null;
        if(trainSeed == null) {
            logger.warning("train seed not set");
        } else {
            trainRandom.setSeed(trainSeed);
        }
        if(tunedFunction != null) {
            tuned = tunedFunction.apply(trainInstances);
        }
    }

    @Override
    public void buildClassifier(Instances trainInstances) throws Exception {
        setupTrain(trainInstances);
        while (tuned.hasNext()) {
            Benchmark benchmark = tuned.next();
            logger.info(tuned.getSelector().getExtractor().apply(benchmark) + " for " + benchmark.getClassifier().toString() + " " + StringUtilities.join(", ", benchmark.getClassifier().getOptions()));
        }
        List<Benchmark> selected = tuned.getSelector().getSelectedAsList();
        logger.info("Best: "); // todo stringbuilder
        for(Benchmark benchmark : selected) {
            logger.info(tuned.getSelector().getExtractor().apply(benchmark) + " for " + benchmark.getClassifier().toString() + " " + StringUtilities.join(", ", benchmark.getClassifier().getOptions()));
        }
        logger.info("Picked: ");
        best = ArrayUtilities.randomChoice(selected, trainRandom);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return best.getClassifier().distributionForInstance(instance);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return best.getClassifier().classifyInstance(instance);
    }

    public Tuned getTuned() {
        return tuned;
    }

    public void setTuned(Tuned tuned) {
        this.tuned = tuned;
    }

    public Logger getLogger() {
        return logger;
    }

    public void setLogger(Logger logger) {
        this.logger = logger;
    }

    public Random getTrainRandom() {
        return trainRandom;
    }

    public void setTrainRandom(Random trainRandom) {
        this.trainRandom = trainRandom;
    }

    @Override
    public void setTestSeed(long seed) {
        testSeed = seed;
    }

    public Long getTrainSeed() {
        return trainSeed;
    }

    @Override
    public Long getTestSeed() {
        return testSeed;
    }

    public void setTrainSeed(long trainSeed) {
        this.trainSeed = trainSeed;
    }

    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV) {

    }

    private String trainEstimatePath;

    @Override
    public void writeTrainEstimatesToFile(String train) {
        trainEstimatePath = train;
    }

    @Override
    public ClassifierResults getTrainResults() {
        return best.getResults();
    }

    @Override
    public String getParameters() {
        return StringUtilities.join(",", best.getClassifier().getOptions());
    }

    @Override
    public void setParamSearch(boolean b) {

    }

    @Override
    public void setParametersFromIndex(int x) {
        index = x;
    }

    @Override
    public String getParas() {
        return null;
    }

    @Override
    public double getAcc() {
        return 0;
    }
}
