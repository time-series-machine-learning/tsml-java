package tsml.classifiers.shapelet_based.dev.classifiers.ensemble;

import evaluation.storage.ClassifierResults;
import tsml.data_containers.TimeSeriesInstance;
import utilities.DebugPrinting;

import java.util.concurrent.TimeUnit;

import static utilities.GenericTools.indexOfMax;

public abstract class ModuleVotingSchemeTS implements DebugPrinting {

    protected int numClasses;
    public boolean needTrainPreds = false;

    public void trainVotingScheme(AbstractEnsembleTS.EnsembleModuleTS[] modules, int numClasses) throws Exception {
        this.numClasses = numClasses;
    }

    public abstract double[] distributionForTrainInstance(AbstractEnsembleTS.EnsembleModuleTS[] modules, int trainInstanceIndex)  throws Exception;

    public double classifyTrainInstance(AbstractEnsembleTS.EnsembleModuleTS[] modules, int trainInstanceIndex) throws Exception {
        double[] dist = distributionForTrainInstance(modules, trainInstanceIndex);
        return indexOfMax(dist);
    }

    public abstract double[] distributionForTestInstance(AbstractEnsembleTS.EnsembleModuleTS[] modules, int testInstanceIndex)  throws Exception;

    public double classifyTestInstance(AbstractEnsembleTS.EnsembleModuleTS[] modules, int testInstanceIndex) throws Exception {
        double[] dist = distributionForTestInstance(modules, testInstanceIndex);
        return indexOfMax(dist);
    }

    public abstract double[] distributionForInstance(AbstractEnsembleTS.EnsembleModuleTS[] modules, TimeSeriesInstance testInstance) throws Exception;

    public double classifyInstance(AbstractEnsembleTS.EnsembleModuleTS[] modules, TimeSeriesInstance testInstance) throws Exception {
        double[] dist = distributionForInstance(modules, testInstance);
        return indexOfMax(dist);
    }

    /**
     * makes array sum to 1
     */
    public double[] normalise(double[] dist) {
        //normalise so all sum to one
        double sum=dist[0];
        for(int i = 1; i < dist.length; i++)
            sum += dist[i];

        if (sum == 0.0)
            for(int i = 0; i < dist.length; i++)
                dist[i] = 1.0/dist.length;
        else
            for(int i = 0; i < dist.length; i++)
                dist[i] /= sum;

        return dist;
    }

    protected double[] distributionForNewInstance(AbstractEnsembleTS.EnsembleModuleTS module, TimeSeriesInstance inst) throws Exception {
        long startTime = System.nanoTime();
        double[] dist = module.getClassifier().distributionForInstance(inst);
        long predTime = System.nanoTime() - startTime;


        return dist;
    }

    public void storeModuleTestResult(AbstractEnsembleTS.EnsembleModuleTS module, double[] dist, long predTime) throws Exception {
        if (module.testResults == null) {
            module.testResults = new ClassifierResults();
            module.testResults.setTimeUnit(TimeUnit.NANOSECONDS);
            module.testResults.setBuildTime(module.trainResults.getBuildTime());
        }

        module.testResults.addPrediction(dist, indexOfMax(dist), predTime, "");
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
}
