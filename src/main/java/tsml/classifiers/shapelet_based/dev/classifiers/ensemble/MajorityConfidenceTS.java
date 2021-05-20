package tsml.classifiers.shapelet_based.dev.classifiers.ensemble;

import tsml.data_containers.TimeSeriesInstance;

public class MajorityConfidenceTS extends ModuleVotingSchemeTS {

    public MajorityConfidenceTS() {
    }

    public MajorityConfidenceTS(int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public void trainVotingScheme(AbstractEnsembleTS.EnsembleModuleTS[] modules, int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public double[] distributionForTrainInstance(AbstractEnsembleTS.EnsembleModuleTS[] modules, int trainInstanceIndex) {
        double[] preds = new double[numClasses];

        for(int m = 0; m < modules.length; m++){
            double[] p=modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex);
            for (int c = 0; c < numClasses; c++) {
                preds[c] += modules[m].priorWeight *
                        modules[m].posteriorWeights[c] * p[c];
            }
        }

        return normalise(preds);
    }

    @Override
    public double[] distributionForTestInstance(AbstractEnsembleTS.EnsembleModuleTS[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];

        for(int m = 0; m < modules.length; m++){
            double[] p=modules[m].testResults.getProbabilityDistribution(testInstanceIndex);
            for (int c = 0; c < numClasses; c++) {
                preds[c] += modules[m].priorWeight *
                        modules[m].posteriorWeights[c] * p[c];
            }
        }

        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(AbstractEnsembleTS.EnsembleModuleTS[] modules, TimeSeriesInstance testInstance) throws Exception {
        double[] preds = new double[numClasses];

        double[] dist;
        for(int m = 0; m < modules.length; m++){
            dist = distributionForNewInstance(modules[m], testInstance);

            for (int c = 0; c < numClasses; c++) {
                preds[c] += modules[m].priorWeight *
                        modules[m].posteriorWeights[c] *
                        dist[c];
            }
        }

        return normalise(preds);
    }

}
