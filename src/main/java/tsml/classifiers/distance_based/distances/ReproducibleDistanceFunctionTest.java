package tsml.classifiers.distance_based.distances;

import static org.junit.Assert.*;

import experiments.data.DatasetLoading;
import gnu.trove.map.TFloatObjectMap;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import weka.core.Debug.Random;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class ReproducibleDistanceFunctionTest {

    protected abstract DistanceFunction getDistanceFunction();

    protected abstract ParamSpace getDistanceFunctionParamSpace();

    protected Instances data;

    @Test
    public void canReproduceKnownDistances() throws Exception {
        final boolean regenerate = true;
        final int numRepeats = 100;

        final double[] distances = new double[numRepeats];
        if(!regenerate) {

        }
        final Instances[] dataset = DatasetLoading.sampleItalyPowerDemand(0);
        data = dataset[0];
        data.addAll(dataset[1]);
        final DistanceFunction distanceFunction = getDistanceFunction();
        final ParamSpace paramSpace = getDistanceFunctionParamSpace();
        for(int i = 0; i < numRepeats; i++) {
            final Random random = new Random(i);
            if(!paramSpace.isEmpty()) {
                ParamSet paramSet = paramSpace.get(random.nextInt(paramSpace.size()));
                distanceFunction.setOptions(paramSet.getOptions());
            }
            Instance a = data.get(random.nextInt(data.size()));
            Instance b = data.get(random.nextInt(data.size()));
            double distance = distanceFunction.distance(a, b);
            if(regenerate) {
                distances[i] = distance;
            } else {
                assertEquals(distances[i], distance, 0);
            }
        }
        if(regenerate) {

        }
    }
}