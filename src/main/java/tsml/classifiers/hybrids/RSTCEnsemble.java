package tsml.classifiers.hybrids;


import evaluation.evaluators.CrossValidationEvaluator;
import machine_learning.classifiers.ensembles.AbstractEnsemble;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import weka.core.Capabilities;

public class RSTCEnsemble extends AbstractEnsemble {

    @Override
    public void setupDefaultEnsembleSettings() {

        this.ensembleName = "HIVE-COTE 0.1";
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();
        this.transform = null;

        CrossValidationEvaluator cv = new CrossValidationEvaluator(seed, false, false, false, false);
        cv.setNumFolds(10);
        this.trainEstimator = cv;

    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // attributes must be numeric
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        // Can only handle discrete class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        // instances
        result.setMinimumNumberInstances(1);
        if(readIndividualsResults)//Can handle all data sets
            result.enableAll();
        return result;
    }

}
