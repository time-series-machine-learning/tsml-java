package tsml.classifiers.distance_based.proximity;

import com.beust.jcommander.internal.Lists;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.utils.collections.PrunedMultimap;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.params.ParamSpace;
import tsml.classifiers.distance_based.utils.params.iteration.RandomSearchIterator;
import tsml.classifiers.distance_based.utils.random.RandomUtils;
import tsml.classifiers.distance_based.utils.scoring.Scorer;
import tsml.classifiers.distance_based.utils.scoring.Scorer.GiniImpurityEntropy;
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

    private boolean earlyAbandonDistances;
    private DistanceFunction distanceFunction;
    private List<List<Instance>> exemplars;
    private boolean randomTieBreakDistances;
    private boolean randomTieBreakR;
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
        setRandomTieBreakDistances(true);
        setRandomTieBreakR(false);
        setEarlyAbandonDistances(false);
        setScorer(new GiniImpurityEntropy());
        setScore(-1);
    }

    private void pickExemplars(final Instances instances) {
//        System.out.println("pe"); todo
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
//        System.out.println("pd"); todo
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
        PrunedMultimap<Double, Container> map = PrunedMultimap.desc();
        if(randomTieBreakR) {
            map.setSoftLimit(1);
        } else {
            map.setHardLimit(1);
        }
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
//                System.out.println("cb: " + j + "," + index); todo
                final Instances closestPartition = partitions.get(index);
                closestPartition.add(instance);
            }
            if(distanceFunction instanceof DistanceMeasureable) {
                ((DistanceMeasureable) distanceFunction).setTraining(false);
            }
            double score = scorer.findScore(data, partitions);
            Container container = new Container(exemplars, distanceFunction, partitions, score);
            map.put(score, container);
//            System.out.println("g: " + (0.5 - score)); // todo
        }
        Container choice = RandomUtils.choice(new ArrayList<>(map.values()), random);
        partitions = choice.partitions;
        distanceFunction = choice.distanceFunction;
        exemplars = choice.exemplars;
        score = choice.score;
//        System.out.println("bg: " + (0.5 - score)); // todo
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
        PrunedMultimap<Double, Integer> distanceToPartitionIndexMap = PrunedMultimap.asc();
        if(randomTieBreakDistances) {
            distanceToPartitionIndexMap.setSoftLimit(1);
        } else {
            distanceToPartitionIndexMap.setHardLimit(1);
        }
        double minDistance = maxDistance;
        for(int i = 0; i < exemplars.size(); i++) {
            for(Instance exemplar : exemplars.get(i)) {
                if(exemplar.equals(instance)) {
                    return i;
                }
                final double distance = distanceFunction.distance(instance, exemplar, limit);
                if(earlyAbandonDistances) {
                    limit = Math.min(distance, limit);
                }
                minDistance = Math.min(distance, minDistance);
                distanceToPartitionIndexMap.put(distance, i);
            }
        }
        final Double smallestDistance = distanceToPartitionIndexMap.firstKey();
        final Collection<Integer> closestPartitionIndices = distanceToPartitionIndexMap.get(smallestDistance);
        // no-op, but must sample the random source to match orig pf. The orig implementation always samples the
        // random irrelevant of whether the list size is 1 or more. We only sample IFF the list size is larger than 1
        // . Therefore we'll sample the random if the list is exactly equal to 1 in size.
        if(closestPartitionIndices.size() == 1) {
            getRandom().nextInt(closestPartitionIndices.size());
        }
        final Integer closestPartitionIndex = Utilities.randPickOne(closestPartitionIndices, getRandom());
        return closestPartitionIndex;
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

    public boolean isEarlyAbandonDistances() {
        return earlyAbandonDistances;
    }

    public void setEarlyAbandonDistances(final boolean earlyAbandonDistances) {
        this.earlyAbandonDistances = earlyAbandonDistances;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public List<List<Instance>> getExemplars() {
        return exemplars;
    }

    private void setExemplars(final List<List<Instance>> exemplars) {
        Assert.assertNotNull(exemplars);
        for(List<Instance> exemplarGroup : exemplars) {
            Assert.assertNotNull(exemplarGroup);
            Assert.assertFalse(exemplarGroup.isEmpty());
        }
        this.exemplars = exemplars;
    }

    private void setDistanceFunction(final DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
    }

    public boolean isRandomTieBreakDistances() {
        return randomTieBreakDistances;
    }

    public void setRandomTieBreakDistances(final boolean randomTieBreakDistances) {
        this.randomTieBreakDistances = randomTieBreakDistances;
    }

    public int getR() {
        return r;
    }

    public void setR(final int r) {
        Assert.assertTrue(r > 0);
        this.r = r;
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

    public boolean isRandomTieBreakR() {
        return randomTieBreakR;
    }

    public void setRandomTieBreakR(final boolean randomTieBreakR) {
        this.randomTieBreakR = randomTieBreakR;
    }

    public DistanceFunctionSpaceBuilder getDistanceFunctionSpaceBuilder() {
        return distanceFunctionSpaceBuilder;
    }

    public ParamSpace getDistanceFunctionSpace() {
        return distanceFunctionSpace;
    }
}
