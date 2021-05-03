package tsml.classifiers.shapelet_based.classifiers.ensemble;

public abstract class ModuleWeightingSchemeTS {

    public boolean uniformWeighting = true;
    public boolean needTrainPreds = true;

    public void defineWeightings(AbstractEnsembleTS.EnsembleModuleTS[] modules, int numClasses) {
        for (AbstractEnsembleTS.EnsembleModuleTS m : modules) //by default, sets weights independently for each module
            m.posteriorWeights = defineWeighting(m, numClasses);
    }

    protected abstract double[] defineWeighting(AbstractEnsembleTS.EnsembleModuleTS trainPredictions, int numClasses);

    protected double[] makeUniformWeighting(double weight, int numClasses) {
        //Prevents all weights from being set to 0 for datasets such as Fungi.
        if (weight == 0) weight = 1;

        double[] weights = new double[numClasses];
        for (int i = 0; i < weights.length; ++i)
            weights[i] = weight;
        return weights;
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }

}