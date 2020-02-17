package tsml.classifiers.distance_based.pf;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainTimeContractable;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class ProximityForest extends EnhancedAbstractClassifier implements TrainTimeContractable {

    private int numTreeLimit = 100;
    private List<ProximityTree> trees;
    private boolean rebuild = true;
    private long trainTimeLimit = -1;


    public ProximityForest() {

    }

    public ProximityForest(ProximityForest other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setRebuild(boolean rebuild) {
        super.setRebuild(rebuild);
        this.rebuild = rebuild;
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        super.buildClassifier(trainData);
        if(rebuild) {
            rebuild = false;
            trees = new ArrayList<>();
        }
        while (hasNext()) {
            next();
        }
    }

    public boolean hasNext() {
        return hasRemainingTraining() && trees.size() < numTreeLimit;
    }

    public ProximityForest next() {
        ProximityTree tree = new ProximityTree();

        trees.add(tree);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }

    public long getTrainTimeLimit() {
        return trainTimeLimit;
    }

    @Override
    public void setTrainTimeLimit(long trainTimeLimit) {
        this.trainTimeLimit = trainTimeLimit;
    }
}
