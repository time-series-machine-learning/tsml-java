package classifiers.template_classifier;

import evaluation.storage.ClassifierResults;
import net.sourceforge.sizeof.SizeOf;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import static utilities.StringUtilities.join;

public abstract class TemplateClassifier
    extends AbstractClassifier
    implements TemplateClassifierInterface {

    private String savePath;
    private long trainContractNanos = -1;
    private Random trainRandom = new Random();
    private long testTimeNanos = -1;
    private Random testRandom = new Random();

    public long remainingTrainContractNanos() {
        if(trainContractNanos < 0) {
            return Long.MAX_VALUE;
        }
        return trainContractNanos - trainTimeNanos;
    }

    protected void setClassifierResultsMetaInfo(ClassifierResults classifierResults) throws
                                                                                     Exception {
        classifierResults.setTestTime(getTestTimeNanos());
        classifierResults.setBuildTime(getTrainTimeNanos());
        classifierResults.setTimeUnit(TimeUnit.NANOSECONDS);
        classifierResults.setParas(join(",", getOptions()));
        try {
            classifierResults.setMemory(SizeOf.deepSizeOf(this));
        } catch (Exception e) {
            classifierResults.setMemory(-1);
        }
    }

    public void incrementTrainTimeNanos(long nanos) {
        trainTimeNanos += nanos;
    }

    public void incrementTestTimeNanos(long nanos) {
        testTimeNanos += nanos;
    }

    public long getTrainTimeNanos() {
        return trainTimeNanos;
    }

    public void setTrainTimeNanos(final long trainTimeNanos) {
        this.trainTimeNanos = trainTimeNanos;
    }

    private long trainTimeNanos = -1;

    public Random getTrainRandom() {
        return trainRandom;
    }

    @Override
    public abstract void buildClassifier(final Instances trainInstances) throws
                                                      Exception;

    @Override
    public void setSavePath(final String path) {
        this.savePath = path;
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        TemplateClassifier other = (TemplateClassifier) obj;
        setSavePath(other.savePath);
        setTrainTimeNanos(other.getTrainTimeNanos());
        setTrainResults(other.getTrainResults());
        if(other.seed != null) {
            setSeed(other.seed);
        }
        setTimeLimit(other.getTrainContractNanos());
    }

    @Override
    public void setTimeLimit(final long time) {
        trainContractNanos = time;
    }

    @Override
    public void setTimeLimit(final TimeLimit time, final int amount) {
        throw new UnsupportedOperationException();
    }

    public void setTrainContractNanos(long nanos) {
        trainContractNanos = nanos;
    }

    public boolean hasTrainContract() {
        return trainContractNanos >= 0;
    }

    public long getTrainContractNanos() {
        return trainContractNanos;
    }

    @Override
    public String getParameters() {
        return join(",", getOptions());
    }

    @Override
    public void setFindTrainAccuracyEstimate(final boolean setCV) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean findsTrainAccuracyEstimate() {
        throw new UnsupportedOperationException();
    }

    private String trainResultsPath;

    @Override
    public void writeCVTrainToFile(final String path) {
        trainResultsPath = path;
    }

    private ClassifierResults trainResults;

    @Override
    public ClassifierResults getTrainResults() {
        return trainResults;
    }

    @Override
    public int setNumberOfFolds(final Instances data) {
        throw new UnsupportedOperationException();
    }

    public void resetTrainRandom() {
        if(seed != null) {
            trainRandom.setSeed(seed);
        }
    }

    public void resetTestRandom() {
        if(seed != null) {
            testRandom.setSeed(seed);
        }
    }

    public void resetRandom() {
        resetTestRandom();
        resetTrainRandom();
    }

    @Override
    public void setSeed(final int seed) {
        this.seed = seed;
        resetRandom();
    }

    @Override
    public int getSeed() {
        if(seed == null) {
            throw new IllegalStateException("seed not set");
        } else {
            return seed;
        }
    }

    private Integer seed = null;

    public String getSavePath() {
        return savePath;
    }

    public long getTestTimeNanos() {
        return testTimeNanos;
    }

    public void setTestTimeNanos(final long testTimeNanos) {
        this.testTimeNanos = testTimeNanos;
    }

    public Random getTestRandom() {
        return testRandom;
    }

    public void setTrainResults(final ClassifierResults trainResults) {
        this.trainResults = trainResults;
    }
}
