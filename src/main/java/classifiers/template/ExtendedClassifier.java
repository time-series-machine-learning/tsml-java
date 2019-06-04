package classifiers.template;

import evaluation.storage.ClassifierResults;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;


import static utilities.StringUtilities.joinWithComma;

public abstract class ExtendedClassifier
    extends AbstractClassifier
    implements ExtendedClassifierInterface {

    private String savePath;
    private long trainContractNanos = -1;

    @Override
    public void setSavePath(final String path) {
        this.savePath = path;
    }

    @Override
    public void copyFromSerObject(final Object obj) throws
                                                    Exception {
        ExtendedClassifier other = (ExtendedClassifier) obj;
        setSavePath(other.savePath);
        setSeed(other.seed);
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
        return joinWithComma(getOptions());
    }

    @Override
    public void setFindTrainAccuracyEstimate(final boolean setCV) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean findsTrainAccuracyEstimate() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeCVTrainToFile(final String train) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ClassifierResults getTrainResults() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int setNumberOfFolds(final Instances data) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setSeed(final int seed) {
        this.seed = seed;
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
}
