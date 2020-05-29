package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import com.beust.jcommander.internal.Lists;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.knn.strategies.RLTunedKNNSetup.ParamSpaceBuilder;
import tsml.classifiers.distance_based.proximity.splitting.scoring.Scorer;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.classifiers.distance_based.utils.params.iteration.RandomSearchIterator;
import tsml.classifiers.distance_based.utils.random.RandomUtils;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: perform a split using several exemplar instances to partition the data based upon proximity.
 * <p>
 * Contributors: goastler
 */
public class ProximitySplit {

    private boolean earlyAbandon;
    private DistanceFunction distanceFunction;
    private List<List<Instance>> exemplars;
    private boolean randomTieBreak;
    private int r;
    private Random random;
    private Scorer scorer;
    private double score;
    private Instances data;
    private List<Instances> partitions;
    private List<DistanceFunctionSpaceBuilder> distanceFunctionSpaceBuilders;
    private DistanceFunctionSpaceBuilder distanceFunctionSpaceBuilder;
    private ParamSpace distanceFunctionSpace;

    public ProximitySplit(Random random) {
        setRandom(random);
        setR(5);
        setRandomTieBreak(false);
        setEarlyAbandon(false);
        setScorer(Scorer.GINI);
        setScore(-1);
    }

    private void pickExemplars(final Instances instances) {
        final Map<Double, Instances> instancesByClass = Utilities.instancesByClass(instances);
        List<List<Instance>> exemplars = Lists.newArrayList(instancesByClass.size());
        for(Double classLabel : instancesByClass.keySet()) {
            final Instances instanceClass = instancesByClass.get(classLabel);
            final Instance exemplar = RandomUtils.choice(instanceClass, getRandom());
            exemplars.add(Lists.newArrayList(exemplar));
        }
        setExemplars(exemplars);
    }

    private void pickDistanceFunction() {
        distanceFunctionSpaceBuilder = RandomUtils.choice(distanceFunctionSpaceBuilders, getRandom());
        distanceFunctionSpace = distanceFunctionSpaceBuilder.build(data);
        RandomSearchIterator iterator = new RandomSearchIterator(getRandom(), distanceFunctionSpace);
        ParamSet paramSet = iterator.next();
        List<Object> list = paramSet.get(DistanceMeasureable.getDistanceFunctionFlag());
        Assert.assertEquals(1, list.size());
        Object obj = list.get(0);
        setDistanceFunction((DistanceFunction) obj);
    }

    public void buildSplit() {
        class Container {
            public final List<List<Instance>> exemplars;
            public final DistanceFunction distanceFunction;
            public final List<Instances> partitions;
            public final double score;

            Container(final List<List<Instance>> exemplars, final DistanceFunction distanceFunction,
                final List<Instances> partitions, final double score) {
                this.exemplars = exemplars;
                this.distanceFunction = distanceFunction;
                this.partitions = partitions;
                this.score = score;
            }
        }
        PrunedMultimap<Double, Container> map = PrunedMultimap.descSoftSingle();
        for(int i = 0; i < r; i++) {
            pickDistanceFunction();
            pickExemplars(data);
            List<Instances> partitions = Lists.newArrayList(exemplars.size());
            if(distanceFunction instanceof DistanceMeasureable) {
                ((DistanceMeasureable) distanceFunction).setTraining(true);
            }
            for(List<Instance> group : exemplars) {
                partitions.add(new Instances(data, 0));
            }
            distanceFunction.setInstances(data);
            for(int j = 0; j < data.size(); j++) {
                final Instance instance = data.get(j);
                final int index = getPartitionIndexFor(instance);
                System.out.println(j + "," + index);
                final Instances closestPartition = partitions.get(index);
                closestPartition.add(instance);
            }
            if(distanceFunction instanceof DistanceMeasureable) {
                ((DistanceMeasureable) distanceFunction).setTraining(false);
            }
            double score = scorer.findScore(data, partitions);
            Container container = new Container(exemplars, distanceFunction, partitions, score);
            map.put(score, container);
        }
        Container choice = RandomUtils.choice(new ArrayList<>(map.values()), random);
        partitions = choice.partitions;
        distanceFunction = choice.distanceFunction;
        exemplars = choice.exemplars;
        score = choice.score;
    }

    public Instances getPartitionFor(Instance instance) {
        final int index = getPartitionIndexFor(instance);
        return partitions.get(index);
    }

    public double[] distributionForInstance(Instance instance) {
        int index = getPartitionIndexFor(instance);
        return distributionForInstance(instance, index);
    }

    public List<Instances> getPartitions() {
        return partitions;
    }

    public Instances getData() {
        return data;
    }

    public void setData(Instances data) {
        Assert.assertNotNull(data);
        this.data = data;
    }

    private void setPartitions(List<Instances> partitions) {
        Assert.assertNotNull(partitions);
        this.partitions = partitions;
    }

    private void setScore(final double score) {
        this.score = score;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("PartitionSet{" +
            "score=" + score +
            ", dataSize=" + data.size());
        if(partitions != null) {
            int i = 0;
            for(Instances instances : partitions) {
                stringBuilder.append(", p" + i + "=" + instances.size());
                i++;
            }
        }
        stringBuilder.append("}");
        return  stringBuilder.toString();
    }

    public Scorer getScorer() {
        return scorer;
    }

    public void setScorer(final Scorer scorer) {
        Assert.assertNotNull(scorer);
        this.scorer = scorer;
    }

    public Random getRandom() {
        return random;
    }

    public void setRandom(final Random random) {
        Assert.assertNotNull(random);
        this.random = random;
    }

    public double getScore() {
        return score;
    }

    public int getPartitionIndexFor(final Instance instance) {
        final double maxDistance = Double.POSITIVE_INFINITY;
        double limit = maxDistance;
        PrunedMultimap<Double, Integer> distanceToPartitionIndexMap = null;
        int best = -1;
        if(randomTieBreak) {
            distanceToPartitionIndexMap = PrunedMultimap.asc();
            distanceToPartitionIndexMap.setSoftLimit(1);
        }
        double minDistance = maxDistance;
        for(int i = 0; i < exemplars.size(); i++) {
            for(Instance exemplar : exemplars.get(i)) {
                final double distance = distanceFunction.distance(instance, exemplar, limit);
                if(earlyAbandon) {
                    limit = Math.min(distance, limit);
                }
                if(randomTieBreak) {
                    minDistance = Math.min(distance, minDistance);
                    distanceToPartitionIndexMap.put(minDistance, i);
                } else {
                    // todo this is the equiv of hard pruned map
                    if(distance < minDistance) {
                        best = i;
                        minDistance = distance;
                    }
                }
            }
        }
        if(randomTieBreak) {
            distanceToPartitionIndexMap.hardPruneToSoftLimit();
            final Double smallestDistance = distanceToPartitionIndexMap.firstKey();
            final Collection<Integer> closestPartitionIndices = distanceToPartitionIndexMap.get(smallestDistance);
            final Integer closestPartitionIndex = Utilities.randPickOne(closestPartitionIndices, getRandom());
            return closestPartitionIndex;
        } else {
            return best;
        }
    }

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

    public boolean isEarlyAbandon() {
        return earlyAbandon;
    }

    public ProximitySplit setEarlyAbandon(final boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
        return this;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public List<List<Instance>> getExemplars() {
        return exemplars;
    }

    private ProximitySplit setExemplars(final List<List<Instance>> exemplars) {
        Assert.assertNotNull(exemplars);
        for(List<Instance> exemplarGroup : exemplars) {
            Assert.assertNotNull(exemplarGroup);
            Assert.assertFalse(exemplarGroup.isEmpty());
        }
        this.exemplars = exemplars;
        return this;
    }

    private ProximitySplit setDistanceFunction(final DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
        return this;
    }

    public boolean isRandomTieBreak() {
        return randomTieBreak;
    }

    public ProximitySplit setRandomTieBreak(final boolean randomTieBreak) {
        this.randomTieBreak = randomTieBreak;
        return this;
    }

    public int getR() {
        return r;
    }

    public ProximitySplit setR(final int r) {
        Assert.assertTrue(r > 0);
        this.r = r;
        return this;
    }

    public List<DistanceFunctionSpaceBuilder> getDistanceFunctionSpaceBuilders() {
        return distanceFunctionSpaceBuilders;
    }

    public void setDistanceFunctionSpaceBuilders(
        final List<DistanceFunctionSpaceBuilder> distanceFunctionSpaceBuilders) {
        Assert.assertNotNull(distanceFunctionSpaceBuilders);
        Assert.assertFalse(distanceFunctionSpaceBuilders.isEmpty());
        this.distanceFunctionSpaceBuilders = distanceFunctionSpaceBuilders;
    }
}
