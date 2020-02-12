package tsml.classifiers.distance_based.interval;

import tsml.classifiers.EnhancedAbstractClassifier;
import utilities.serialisation.SerConsumer;
import weka.core.Instances;

import java.util.List;

public class ProximityTree extends EnhancedAbstractClassifier {
    // todo estimate train const / actually do it


    private Splitter splitter = data -> {
        throw new UnsupportedOperationException();
    }
    private boolean retrain = true;
    private SerConsumer<Instances> trainSetupFunction = (SerConsumer<Instances>) instances -> {};

    @Override public void setRetrain(final boolean retrain) {
        this.retrain = retrain;
        super.setRetrain(retrain);
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        super.buildClassifier(trainData);
        if(retrain) {
            trainSetupFunction.accept(trainData);
        }
    }

    public Splitter getSplitter() {
        return splitter;
    }

    public void setSplitter(final Splitter splitter) {
        this.splitter = splitter;
    }

    public SerConsumer<Instances> getTrainSetupFunction() {
        return trainSetupFunction;
    }

    public void setTrainSetupFunction(final SerConsumer<Instances> trainSetupFunction) {
        this.trainSetupFunction = trainSetupFunction;
    }

    public static void main(String[] args) {
        ProximityTree pt = new ProximityTree();
        pt.setTrainSetupFunction(trainData -> {

        });
    }
}
