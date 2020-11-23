package tsml.classifiers.distance_based.utils.experiment;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.Instances;

public class DatasetSplit {
    private Instances trainDataArff;
    private Instances testDataArff;
    private TimeSeriesInstances trainDataTs;
    private TimeSeriesInstances testDataTs;

    public DatasetSplit(TimeSeriesInstances trainData, TimeSeriesInstances testData) {
        setTrainData(trainData);
        setTestData(testData);
    }
    
    public DatasetSplit(Instances trainData, Instances testData) {
        setTrainData(trainData);
        setTestData(testData);
    }
    
    public DatasetSplit(Instances[] data) {
        this(data[0], data[1]);
    }
    
    public Instances getTestDataArff() {
        if(testDataArff == null) {
            setTestData(Converter.toArff(testDataTs));
        }
        return testDataArff;
    }
    
    public TimeSeriesInstances getTestDataTS() {
        if(testDataTs == null) {
            setTestData(Converter.fromArff(testDataArff));
        }
        return testDataTs;
    }

    public void setTestData(final Instances testData) {
        this.testDataArff = testData;
        testDataTs = null;
    }
    
    public void setTestData(final TimeSeriesInstances testData) {
        this.testDataTs = testData;
        testDataArff = null;
    }

    public Instances getTrainDataArff() {
        if(trainDataArff == null) {
            setTrainData(Converter.toArff(trainDataTs));
        }
        return trainDataArff;
    }
    
    public TimeSeriesInstances getTrainDataTS() {
        if(trainDataTs == null) {
            setTrainData(Converter.fromArff(trainDataArff));
        }
        return trainDataTs;
    }

    public void setTrainData(final Instances trainData) {
        this.trainDataArff = trainData;
        trainDataTs = null;
    }

    public void setTrainData(final TimeSeriesInstances trainData) {
        this.trainDataTs = trainData;
        trainDataArff = null;
    }
}
