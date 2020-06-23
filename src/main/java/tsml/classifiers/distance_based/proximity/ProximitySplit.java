package tsml.classifiers.distance_based.proximity;

import com.beust.jcommander.internal.Lists;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.EnumBasedClassifierConfigurer;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap.DiscardType;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearchIterator;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.stats.scoring.PartitionScorer;
import tsml.classifiers.distance_based.utils.stats.scoring.PartitionScorer.GiniImpurityEntropy;
import tsml.transformers.Indexer;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Purpose: perform a split using several exemplar instances to partition the data based upon proximity.
 * <p>
 * Contributors: goastler
 */
public class ProximitySplit extends BaseClassifier {

    // the various configs for this classifier
    public enum Config implements EnumBasedClassifierConfigurer<ProximitySplit> {
        R1() {
            @Override
            public <B extends ProximitySplit> B applyConfigTo(B proximitySplit) {
                proximitySplit = super.applyConfigTo(proximitySplit);
                proximitySplit.setRandomTieBreakDistances(true);
                proximitySplit.setRandomTieBreakCandidates(false);
                proximitySplit.setEarlyAbandonDistances(false);
                proximitySplit.setPartitionScorer(new GiniImpurityEntropy());
                proximitySplit.setScore(-1);
                proximitySplit.setRandomR(false);
                proximitySplit.setExemplarCheckOriginal(true);
                proximitySplit.setMatchOriginalPFRandomCalls(false);
                proximitySplit.setR(1);
                return proximitySplit;
            }
        },
        R5() {
            @Override
            public <B extends ProximitySplit> B applyConfigTo(B proximitySplit) {
                proximitySplit = R1.applyConfigTo(proximitySplit);
                proximitySplit = super.applyConfigTo(proximitySplit);
                proximitySplit.setR(5);
                return proximitySplit;
            }
        },
        R10() {
            @Override
            public <B extends ProximitySplit> B applyConfigTo(B proximitySplit) {
                proximitySplit = R1.applyConfigTo(proximitySplit);
                proximitySplit = super.applyConfigTo(proximitySplit);
                proximitySplit.setR(10);
                return proximitySplit;
            }
        },
        RR5() {
            @Override
            public <B extends ProximitySplit> B applyConfigTo(B proximitySplit) {
                proximitySplit = R5.applyConfigTo(proximitySplit);
                proximitySplit = super.applyConfigTo(proximitySplit);
                proximitySplit.setRandomR(true);
                return proximitySplit;
            }
        },
        RR10() {
            @Override
            public <B extends ProximitySplit> B applyConfigTo(B proximitySplit) {
                proximitySplit = R10.applyConfigTo(proximitySplit);
                proximitySplit = super.applyConfigTo(proximitySplit);
                proximitySplit.setRandomR(true);
                return proximitySplit;
            }
        }
    }

    /**
     * @param random the random source
     */
    public ProximitySplit(Random random) {
        setRandom(random);
        Config.R5.applyConfigTo(this);
    }

    // whether to early abandon distance measurements for distance between instances (data) and exemplars
    private boolean earlyAbandonDistances;
    // the distance function for comparing instances to exemplars
    private DistanceFunction distanceFunction;
    // the groups of exemplars. Each sub list is a group of exemplars. Each group becomes a branch eventually.
    // Instances are classified based on their proximity to the nearest group of exemplars
    private List<List<Instance>> exemplarGroups;
    // whether to random tie break distances (e.g. exemplar A and B have a distance of 3.5 to instance X, which to
    // choose?)
    private boolean randomTieBreakDistances;
    // whether to random tie break R. R splits are considered during building this split. If multiple splits have the
    // same score, which should be chosen?
    private boolean randomTieBreakCandidates;
    // the number of splits to consider for this split
    private int r;
    // whether to choose the number of splits randomly
    private boolean randomR; // todo random r
    // the max number of splits to be considered. This only matters when using randomR
    private int maxR;
    // a method of scoring the split of data into partitions
    private PartitionScorer partitionScorer;
    // the score of this split
    private double score;
    // the data at this split (i.e. before being partitioned)
    private Instances data;
    // the partitions of the data after being split
    private List<Instances> partitions;
    // the distance function space builders. Several distance function spaces depend on the input data to compute
    // parameters. These builders take a set of data and compute the spaces from the data
    private List<ParamSpaceBuilder> distanceFunctionSpaceBuilders;
    // the distance function space builder chosen for this split
    private ParamSpaceBuilder distanceFunctionSpaceBuilder;
    // the distance function space chosen for this split
    private ParamSpace distanceFunctionSpace;
    // whether to match the original Proximity Forest results exactly. This is only useful if mirroring PF parameters
    // exactly
    private boolean matchOriginalPFRandomCalls;
    // whether to check for exemplar matching inside the loop (original) or before any looping (improved method)
    private boolean exemplarCheckOriginal;

    /**
     * build the split using the data provided
     */
    public void buildClassifier() throws Exception {
        buildClassifier(data);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        super.buildClassifier(trainData);
        Indexer.index(trainData);
        // small helper class to contain a split. This is used to temporarily hold split results while this function
        // compares R splits to pick the best
        class SplitCandididate {

            SplitCandididate(final List<List<Instance>> exemplars, final DistanceFunction distanceFunction,
                             final List<Instances> partitions, final double score) {
                this.exemplars = exemplars;
                this.distanceFunction = distanceFunction;
                this.partitions = partitions;
                this.score = score;
            }

            public final List<List<Instance>> exemplars;
            public final DistanceFunction distanceFunction;
            public final List<Instances> partitions;
            public final double score;
        }
        // change the view of the data into per class
        final Map<Double, Instances> instancesByClass = Utilities.instancesByClass(data);
        // make a map to store the best X splits
        final PrunedMultimap<Double, SplitCandididate> map = PrunedMultimap.desc();
        if(randomTieBreakCandidates) {
            // splits which score the same will be kept, then random choice between them
            map.setSoftLimit(1);
        } else {
            // splits which score the same will not be kept, only 1 will be kept at any point in time
            map.setHardLimit(1);
            // discard the youngest split. i.e. if there's 2 splits which score 0.4, the newest split will be discarded
            map.setDiscardType(DiscardType.NEWEST);
        }
        // randomly set R if enabled
        if(randomR) {
            maxR = r;
            r = rand.nextInt(maxR + 1) + 1;
        }
        // for every split attempt
        for(int i = 0; i < r; i++) {
            // pick the distance function
            pickDistanceFunction();
            // pick the exemplars
            pickExemplars(instancesByClass);
            // setup the partitions
            List<Instances> partitions = Lists.newArrayList(exemplarGroups.size());
            if(distanceFunction instanceof DistanceMeasureable) {
                ((DistanceMeasureable) distanceFunction).setTraining(true);
            }
            for(List<Instance> group : exemplarGroups) {
                partitions.add(new Instances(data, 0));
            }
            distanceFunction.setInstances(data);
            // go through every instance and find which partition it should go into. This should be the partition
            // with the closest exemplar associate
            for(int j = 0; j < data.size(); j++) {
                final Instance instance = data.get(j);
                final int index = getPartitionIndexFor(instance);
                final Instances closestPartition = partitions.get(index);
                closestPartition.add(instance);
            }
            if(distanceFunction instanceof DistanceMeasureable) {
                ((DistanceMeasureable) distanceFunction).setTraining(false);
            }
            // find the score of this split attempt, i.e. how good it is
            double score = partitionScorer.findScore(data, partitions);
            // chuck into a container to keep for later
            SplitCandididate splitCandididate = new SplitCandididate(exemplarGroups, distanceFunction, partitions,
                                                                     score);
            // add it to the map. The map will handle whether the split attempt was any good and should be kept
            map.put(score, splitCandididate);
        }
        // choose the best of the R splits. The map handles the tie break if necessary
        SplitCandididate choice = RandomUtils.choice(new ArrayList<>(map.values()), rand);
        // populate the fields of this split from the split attempt
        setPartitions(choice.partitions);
        setDistanceFunction(choice.distanceFunction);
        setExemplarGroups(choice.exemplars);
        setScore(choice.score);
        ResultUtils.setInfo(trainResults, this, trainData);
    }

    /**
     * pick the distance function
     */
    private void pickDistanceFunction() {
        // pick a random space
        distanceFunctionSpaceBuilder = RandomUtils.choice(distanceFunctionSpaceBuilders, getRandom());
        // built that space
        distanceFunctionSpace = distanceFunctionSpaceBuilder.build(data);
        // randomly pick the distance function / parameters from that space
        RandomSearchIterator iterator = new RandomSearchIterator(getRandom(), distanceFunctionSpace);
        ParamSet paramSet = iterator.next();
        // there is only one distance function in the ParamSet returned
        List<Object> list = paramSet.get(DistanceMeasureable.DISTANCE_MEASURE_FLAG);
        Assert.assertEquals(1, list.size());
        // cast it into shape
        Object obj = list.get(0);
        setDistanceFunction((DistanceFunction) obj);
    }

    /**
     * pick exemplars from the given dataset
     *
     * @param instancesByClass a map of class labels to instances
     */
    private void pickExemplars(final Map<Double, Instances> instancesByClass) {
        // pick one exemplar per class
        List<List<Instance>> exemplars = Lists.newArrayList(instancesByClass.size());
        for(Double classLabel : instancesByClass.keySet()) {
            final Instances instanceClass = instancesByClass.get(classLabel);
            final Instance exemplar = RandomUtils.choice(instanceClass, getRandom());
            // orig pf always calls for a random number even if the instances has only one instance. The choice
            // function doesn't source a random number given a list of size 1, therefore the random number call must
            // be made here to match orig pf random number sequence
            if(instanceClass.size() == 1) {
                getRandom().nextInt(1);
            }
            exemplars.add(Lists.newArrayList(exemplar));
        }
        setExemplarGroups(exemplars);
    }

    /**
     * get the partition index of the given instance
     *
     * @param instance
     * @return
     */
    public int getPartitionIndexFor(final Instance instance) {
        // check the instance isn't an exemplar. If it is, it must belong to the associated partition. No need for
        // distance measurements at all
        if(!exemplarCheckOriginal) {
            // for each exemplar group
            for(int i = 0; i < exemplarGroups.size(); i++) {
                final List<Instance> group = exemplarGroups.get(i);
                // for each exemplar
                for(Instance exemplar : group) {
                    if(exemplar.equals(instance)) {
                        return i;
                    }
                }
            }
        }
        // the limit for early abandon
        double limit = Double.POSITIVE_INFINITY;
        // a map to maintain the closest partition indices
        PrunedMultimap<Double, Integer> distanceToPartitionIndexMap = PrunedMultimap.asc();
        if(randomTieBreakDistances) {
            // let the map keep all ties and randomly choose at the end
            distanceToPartitionIndexMap.setSoftLimit(1);
        } else {
            // only keep 1 partition at any point in time, even if multiple partitions are equally close
            distanceToPartitionIndexMap.setHardLimit(1);
            // discard the newest on tie break situation
            distanceToPartitionIndexMap.setDiscardType(DiscardType.NEWEST);
        }
        // loop through exemplar groups
        for(int i = 0; i < exemplarGroups.size(); i++) {
            // for each exemplar in the current group
            for(Instance exemplar : exemplarGroups.get(i)) {
                // check the instance isn't an exemplar
                if(exemplarCheckOriginal && exemplar.equals(instance)) {
                    return i;
                }
                // find the distance
                final double distance = distanceFunction.distance(instance, exemplar, limit);
                // adjust early abandon
                if(earlyAbandonDistances) {
                    limit = Math.min(distance, limit);
                }
                // add the distance and partition index to the map
                distanceToPartitionIndexMap.put(distance, i);
            }
        }
        // get the smallest distance from the map
        final Double smallestDistance = distanceToPartitionIndexMap.firstKey();
        // find the list of corresponding partition indices
        final List<Integer> closestPartitionIndices = distanceToPartitionIndexMap.get(smallestDistance);
        final int numIndices = closestPartitionIndices.size();
        if(!randomTieBreakDistances) {
            Assert.assertEquals(numIndices, 1);
        }
        // random pick the best index
        final Integer closestPartitionIndex = Utilities.randPickOne(closestPartitionIndices, getRandom());
        // no-op, but must sample the random source to match orig pf. The orig implementation always samples the
        // random irrelevant of whether the list size is 1 or more. We only sample IFF the list size is larger than 1
        // . Therefore we'll sample the random if the list is exactly equal to 1 in size.
        if(matchOriginalPFRandomCalls && numIndices == 1) {
            getRandom().nextInt(numIndices);
        }
        return closestPartitionIndex;
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        int index = getPartitionIndexFor(instance);
        return distributionForInstance(instance, index);
    }

    /**
     * find the distribution for the given instance and partition index it belongs to
     *
     * @param instance the instance
     * @param index    the index of the partition it belongs to
     * @return the distribution
     */
    public double[] distributionForInstance(final Instance instance, int index) {
        // this is a simple majority vote over all the exemplars in the exemplars group at the given partition
        // get the corresponding closest exemplars
        final List<Instance> exemplars = exemplarGroups.get(index);
        final double[] distribution = new double[instance.numClasses()];
        // for each exemplar
        for(Instance exemplar : exemplars) {
            // vote for the exemplar's class
            double classValue = exemplar.classValue();
            distribution[(int) classValue]++;
        }
        ArrayUtilities.normaliseInPlace(distribution);
        return distribution;
    }

    public Instances getPartitionFor(Instance instance) {
        final int index = getPartitionIndexFor(instance);
        return partitions.get(index);
    }

    public List<Instances> getPartitions() {
        return partitions;
    }

    private void setPartitions(List<Instances> partitions) {
        Assert.assertNotNull(partitions);
        this.partitions = partitions;
    }

    public Instances getData() {
        return data;
    }

    public void setData(Instances data) {
        Assert.assertNotNull(data);
        this.data = data;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(getClass().getSimpleName() + "{" +
                             "score=" + score +
                             ", dataSize=" + data.size());
        stringBuilder.append(", df=");
        if(distanceFunction != null) {
            stringBuilder.append(distanceFunction.toString());
        } else {
            stringBuilder.append("null");
        }
        if(partitions != null) {
            int i = 0;
            for(Instances instances : partitions) {
                stringBuilder.append(", p" + i + "=" + instances.size());
                i++;
            }
        }
        stringBuilder.append("}");
        return stringBuilder.toString();
    }

    public PartitionScorer getPartitionScorer() {
        return partitionScorer;
    }

    public void setPartitionScorer(final PartitionScorer partitionScorer) {
        Assert.assertNotNull(partitionScorer);
        this.partitionScorer = partitionScorer;
    }

    public double getScore() {
        return score;
    }

    protected void setScore(final double score) {
        this.score = score;
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

    private void setDistanceFunction(final DistanceFunction distanceFunction) {
        Assert.assertNotNull(distanceFunction);
        this.distanceFunction = distanceFunction;
    }

    public List<List<Instance>> getExemplarGroups() {
        return exemplarGroups;
    }

    private void setExemplarGroups(final List<List<Instance>> exemplarGroups) {
        Assert.assertNotNull(exemplarGroups);
        for(List<Instance> exemplarGroup : exemplarGroups) {
            Assert.assertNotNull(exemplarGroup);
            Assert.assertFalse(exemplarGroup.isEmpty());
        }
        this.exemplarGroups = exemplarGroups;
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

    public List<ParamSpaceBuilder> getDistanceFunctionSpaceBuilders() {
        return distanceFunctionSpaceBuilders;
    }

    public void setDistanceFunctionSpaceBuilders(
            final List<ParamSpaceBuilder> distanceFunctionSpaceBuilders) {
        Assert.assertNotNull(distanceFunctionSpaceBuilders);
        Assert.assertFalse(distanceFunctionSpaceBuilders.isEmpty());
        this.distanceFunctionSpaceBuilders = distanceFunctionSpaceBuilders;
    }

    public boolean isRandomTieBreakCandidates() {
        return randomTieBreakCandidates;
    }

    public void setRandomTieBreakCandidates(final boolean randomTieBreakCandidates) {
        this.randomTieBreakCandidates = randomTieBreakCandidates;
    }

    public ParamSpaceBuilder getDistanceFunctionSpaceBuilder() {
        return distanceFunctionSpaceBuilder;
    }

    public ParamSpace getDistanceFunctionSpace() {
        return distanceFunctionSpace;
    }

    public boolean isRandomR() {
        return randomR;
    }

    public void setRandomR(final boolean randomR) {
        this.randomR = randomR;
    }

    public int getMaxR() {
        return maxR;
    }

    public boolean isMatchOriginalPFRandomCalls() {
        return matchOriginalPFRandomCalls;
    }

    public void setMatchOriginalPFRandomCalls(final boolean matchOriginalPFRandomCalls) {
        this.matchOriginalPFRandomCalls = matchOriginalPFRandomCalls;
    }

    public void setExemplarCheckOriginal(final boolean exemplarCheckOriginal) {
        this.exemplarCheckOriginal = exemplarCheckOriginal;
    }
}
