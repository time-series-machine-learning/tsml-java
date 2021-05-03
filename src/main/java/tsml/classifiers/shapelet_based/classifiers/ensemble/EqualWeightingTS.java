package tsml.classifiers.shapelet_based.classifiers.ensemble;

public class EqualWeightingTS extends ModuleWeightingSchemeTS {

    public EqualWeightingTS() {
        uniformWeighting = true;
        needTrainPreds = false;
    }

    @Override
    public double[] defineWeighting(AbstractEnsembleTS.EnsembleModuleTS trainPredictions, int numClasses) {
        return makeUniformWeighting(1.0, numClasses);
    }

}
