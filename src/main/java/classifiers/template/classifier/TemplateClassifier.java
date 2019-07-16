package classifiers.template.classifier;

import evaluation.storage.ClassifierResults;
import net.sourceforge.sizeof.SizeOf;
import utilities.ArrayUtilities;
import utilities.StopWatch;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static utilities.ArrayUtilities.argMax;
import static utilities.StringUtilities.join;

public abstract class TemplateClassifier
    extends AbstractClassifier
    implements TemplateClassifierInterface {

    public TemplateClassifier(final Object other) throws
                                             Exception {
        // copy constructor
        copyFrom(other);
    }

    public TemplateClassifier() {
        // default constructor
    }

    @Override
    public abstract TemplateClassifier copy() throws
                                              Exception;

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    private String savePath;
    private long trainContractNanos = -1;
    private Random trainRandom = new Random();
    private Random testRandom = new Random();
    private Integer testInstancesHash = null;
    private Integer trainInstancesHash = null;
    private StopWatch trainStopWatch = new StopWatch();
    private StopWatch testStopWatch = new StopWatch();

    public abstract void setOption(String key, String value) throws Exception;

    public final void setOptions(String[] options) throws
                                             Exception {
        if(options.length % 2 != 0) {
            throw new IllegalArgumentException("options is not correct length, must be key-value pairs");
        }
        for(int i = 0; i < options.length; i += 2) {
            setOption(options[i], options[i + 1]);
        }
    }

    public String[] getOptions() {
        // todo!
        return new String[0];
    }

    protected boolean trainSetChanged(Instances trainInstances) {
        int hash = trainInstances.hashCode();
        if (trainInstancesHash == null || hash != trainInstancesHash) {
            trainInstancesHash = hash;
            return true;
        } else {
            return false;
        }
    }

    @Override
    public double classifyInstance(final Instance instance) throws
                                                            Exception {
        return ArrayUtilities.indexOfMax(distributionForInstance(instance), getTestRandom());
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

    private Long trainSeed = null;

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
        copyFrom(obj);
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

    private Long testSeed = null;

    public void copyFrom(Object object) throws
                                        Exception {
        TemplateClassifier other = (TemplateClassifier) object;
        setSavePath(other.savePath);
        trainStopWatch = other.trainStopWatch;
        testStopWatch = other.testStopWatch;
        setTrainResults(other.getTrainResults());
        trainSeed = other.trainSeed;
        testSeed = other.testSeed;
        setTimeLimit(other.getTrainContractNanos());
        testInstancesHash = other.testInstancesHash;
        trainInstancesHash = other.trainInstancesHash;
    }

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

    public void resetTrainSeed() {
        // todo null check
        setTrainSeed(getTrainSeed());
    }

    @Override
    public Long getTestSeed() {
        return testSeed;
    }

    public void resetTestSeed() {
        // todo null check
        setTestSeed(getTestSeed());
    }

    public void setTestSeed(final Long testSeed) {
        this.testSeed = testSeed;
    }

    @Override
    public Long getTrainSeed() {
        return trainSeed;
    }

    public void setTrainSeed(final Long trainSeed) {
        this.trainSeed = trainSeed;
    }




}
