package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import com.beust.jcommander.internal.Lists;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.transformed.TransformedDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.TransformedDistanceMeasureable;
import tsml.classifiers.distance_based.proximity.splitting.Split;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import tsml.filters.CachedFilter;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

/**
 * Purpose: perform a split using several exemplar instances to partition the data based upon proximity.
 * <p>
 * Contributors: goastler
 */
public class RandomExemplarProximitySplit extends Split {

    private ExemplarPicker exemplarPicker;
    private boolean useEarlyAbandon = true;
    private DistanceFunction distanceFunction;
    private List<List<Instance>> exemplars;
    private DistanceFunctionPicker distanceFunctionPicker;

    public RandomExemplarProximitySplit(Random random,
        ExemplarPicker exemplarPicker, DistanceFunctionPicker distanceFunctionPicker) {
        super(random);
        setExemplarPicker(exemplarPicker);
        setDistanceFunctionPicker(distanceFunctionPicker);
    }

    /**
     * pick exemplars using the picker. Find a random distance measure.
     * @param data
     * @return
     */
    @Override
    public List<Instances> performSplit(Instances data) {
        List<List<Instance>> exemplars = exemplarPicker.pickExemplars(data);
        setExemplars(exemplars);
        List<Instances> partitions = Lists.newArrayList(exemplars.size());
        DistanceFunction distanceFunction = distanceFunctionPicker.pickDistanceFunction();
        setDistanceFunction(distanceFunction);
        if(distanceFunction instanceof DistanceMeasureable) {
            ((DistanceMeasureable) distanceFunction).setTraining(true);
        }
        for(List<Instance> group : exemplars) {
            partitions.add(new Instances(data, 0));
        }
        distanceFunction.setInstances(data);
        for(Instance instance : data) {
            final int index = getPartitionIndexFor(instance);
            final Instances closestPartition = partitions.get(index);
            closestPartition.add(instance);
        }
        if(distanceFunction instanceof DistanceMeasureable) {
            ((DistanceMeasureable) distanceFunction).setTraining(false);
        }
        return partitions;
    }

    public int getPartitionIndexFor(final Instance instance) {
        double maxDistance = Double.POSITIVE_INFINITY;
        double limit = maxDistance;
//        final PrunedMultimap<Double, Integer> distanceToPartitionIndexMap = PrunedMultimap.asc();
//        distanceToPartitionIndexMap.setSoftLimit(1);
        final boolean useEarlyAbandon = isUseEarlyAbandon();
        int best = -1;
        for(int i = 0; i < exemplars.size(); i++) {
            double minDistance = maxDistance;
            for(Instance exemplar : exemplars.get(i)) {
                final double distance = distanceFunction.distance(instance, exemplar, limit);
                if(useEarlyAbandon) {
                    limit = Math.min(distance, limit);
                }
                if(distance < minDistance) {
                    best = i;
                    minDistance = distance;
                }
//                minDistance = Math.min(distance, minDistance);
            }
//            distanceToPartitionIndexMap.put(minDistance, i);
        }
        return best;
//        distanceToPartitionIndexMap.hardPruneToSoftLimit();
//        final Double smallestDistance = distanceToPartitionIndexMap.firstKey();
//        final Collection<Integer> closestPartitionIndices = distanceToPartitionIndexMap.get(smallestDistance);
//        final Integer closestPartitionIndex = Utilities.randPickOne(closestPartitionIndices, getRandom());
//        return closestPartitionIndex;
    }

    @Override
    public double[] distributionForInstance(final Instance instance, int index) {
        // get the corresponding closest exemplars
        List<Instance> exemplars = this.exemplars.get(index);
        double[] distribution = new double[instance.numClasses()];
        // for each exemplar
        for(Instance exemplar : exemplars) {
            // vote for the exemplar's class
            double classValue = exemplar.classValue();
            distribution[(int) classValue]++;
        }
        ArrayUtilities.normaliseInPlace(distribution);
        return distribution;
    }

    public ExemplarPicker getExemplarPicker() {
        return exemplarPicker;
    }

    public RandomExemplarProximitySplit setExemplarPicker(
        ExemplarPicker exemplarPicker) {
        Assert.assertNotNull(exemplarPicker);
        this.exemplarPicker = exemplarPicker;
        return this;
    }

    public boolean isUseEarlyAbandon() {
        return useEarlyAbandon;
    }

    public RandomExemplarProximitySplit setUseEarlyAbandon(final boolean useEarlyAbandon) {
        this.useEarlyAbandon = useEarlyAbandon;
        return this;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public List<List<Instance>> getExemplars() {
        return exemplars;
    }

    protected RandomExemplarProximitySplit setExemplars(final List<List<Instance>> exemplars) {
        Assert.assertNotNull(exemplars);
        for(List<Instance> exemplarGroup : exemplars) {
            Assert.assertNotNull(exemplarGroup);
            Assert.assertFalse(exemplarGroup.isEmpty());
        }
        this.exemplars = exemplars;
        return this;
    }

    protected RandomExemplarProximitySplit setDistanceFunction(final DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
        return this;
    }

    public DistanceFunctionPicker getDistanceFunctionPicker() {
        return distanceFunctionPicker;
    }

    public RandomExemplarProximitySplit setDistanceFunctionPicker(
        final DistanceFunctionPicker distanceFunctionPicker) {
        Assert.assertNotNull(distanceFunctionPicker);
        this.distanceFunctionPicker = distanceFunctionPicker;
        return this;
    }
}
