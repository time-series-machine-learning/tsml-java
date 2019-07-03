package classifiers.template_classifier;

import evaluation.storage.ClassifierResults;
import net.sourceforge.sizeof.SizeOf;
import utilities.StopWatch;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static utilities.ArrayUtilities.argMax;
import static utilities.StringUtilities.join;

public abstract class TemplateClassifier
    extends AbstractClassifier
    implements TemplateClassifierInterface {

    private String savePath;
    private long trainContractNanos = -1;
    private Random trainRandom = new Random();
    private Random testRandom = new Random();
    private Integer testInstancesHash = null;
    private Integer trainInstancesHash = null;
    private StopWatch trainStopWatch = new StopWatch();
    private StopWatch testStopWatch = new StopWatch();

    protected boolean trainSetChanged(Instances trainInstances) {
        int hash = trainInstances.hashCode();
        if (trainInstancesHash == null || hash != trainInstancesHash) {
            trainInstancesHash = hash;
            return true;
        } else {
            return false;
        }
    }

    protected boolean testSetChanged(Instances testInstances) {
        int hash = testInstances.hashCode();
        if (testInstancesHash == null || hash != testInstancesHash) {
            testInstancesHash = hash;
            return true;
        } else {
            return false;
        }
    }

    public long remainingTrainContractNanos() {
        if(trainContractNanos < 0) {
            return Long.MAX_VALUE;
        }
        return trainContractNanos - trainStopWatch.get();
    }

    protected void setClassifierResultsMetaInfo(ClassifierResults classifierResults) throws
                                                                                     Exception {
        classifierResults.setTestTime(testStopWatch.get());
        classifierResults.setBuildTime(trainStopWatch.get());
        classifierResults.setTimeUnit(TimeUnit.NANOSECONDS);
        classifierResults.setParas(join(",", getOptions()));
        try {
            classifierResults.setMemory(SizeOf.deepSizeOf(this));
        } catch (Exception e) {
            classifierResults.setMemory(-1);
        }
    }

    public Random getTrainRandom() {
        return trainRandom;
    }

    public void copyFrom(Object object) throws
                                        Exception {
        copyFromSerObject(object);
    }

    public boolean withinTrainContract() {
        return remainingTrainContractNanos() > 0;
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
        trainStopWatch = other.trainStopWatch;
        testStopWatch = other.testStopWatch;
        setTrainResults(other.getTrainResults());
        if(other.seed != null) {
            setSeed(other.seed);
        }
        setTimeLimit(other.getTrainContractNanos());
        testInstancesHash = other.testInstancesHash;
        trainInstancesHash = other.trainInstancesHash;
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

    public Random getTestRandom() {
        return testRandom;
    }

    public void setTrainResults(final ClassifierResults trainResults) {
        this.trainResults = trainResults;
    }

    // todo use test stopwatch below
    public ClassifierResults getTestResults(Instances testInstances) throws
                                                                     Exception {
        ClassifierResults results = new ClassifierResults();
        for(Instance testInstance : testInstances) {
            long time = System.nanoTime();
            double[] distribution = distributionForInstance(testInstance);
            time = System.nanoTime() - time;
            results.addPrediction(testInstance.classValue(), distribution, argMax(distribution), time, null);
        }
        setClassifierResultsMetaInfo(results);
        return results;
    }

    public StopWatch getTestStopWatch() {
        return testStopWatch;
    }

    public StopWatch getTrainStopWatch() {
        return trainStopWatch;
    }


    public String getTrainResultsPath() {
        return trainResultsPath;
    }
}
