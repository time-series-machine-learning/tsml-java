package tsml.classifiers.distance_based.proximity;

import com.beust.jcommander.internal.Lists;
import org.junit.Assert;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.BaseTransformDistanceMeasure;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.classifiers.EnumBasedConfigurer;
import tsml.classifiers.distance_based.utils.collections.intervals.Interval;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap.DiscardType;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.RandomSearchIterator;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.stats.scoring.PartitionScorer;
import tsml.classifiers.distance_based.utils.stats.scoring.GiniEntropy;
import tsml.transformers.IntervalTransform;
import tsml.transformers.TransformPipeline;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Purpose: perform a split using several exemplar instances to partition the data based upon proximity.
 * <p>
 * Contributors: goastler
 */
public class ProximitySplit extends BaseClassifier {

    // the various configs for this classifier
    public enum Config implements EnumBasedConfigurer<ProximitySplit> {
        PS_R1() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit.setRandomTieBreakDistances(true);
                proximitySplit.setRandomTieBreakCandidates(false);
                proximitySplit.setEarlyAbandonDistances(false);
                proximitySplit.setPartitionScorer(new GiniEntropy());
                proximitySplit.setReduceSplitTestSize(false);
                proximitySplit.setScore(-1);
                proximitySplit.setRandomR(false);
                proximitySplit.setExemplarCheckOriginal(true);
                proximitySplit.setMatchOriginalPFRandomCalls(false);
                proximitySplit.setR(1);
                proximitySplit.setMaxR(-1);
                proximitySplit.setIntervalTransform(null);
                proximitySplit.setRandomIntervals(false);
                proximitySplit.setMinIntervalSize(-1);
                return proximitySplit;
            }
        },
        PS_R5_SQUICK() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R5.configure(proximitySplit);
                proximitySplit.setReduceSplitTestSize(true);
                return proximitySplit;
            }
        },
        PS_R1_SQUICK() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R1.configure(proximitySplit);
                proximitySplit.setReduceSplitTestSize(true);
                return proximitySplit;
            }
        },
        PS_R10_SQUICK() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R10.configure(proximitySplit);
                proximitySplit.setReduceSplitTestSize(true);
                return proximitySplit;
            }
        },
        PS_R5_MQUICK() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R5.configure(proximitySplit);
                proximitySplit.setReduceSplitTestSize(true);
                proximitySplit.setEarlyAbandonDistances(true);
                proximitySplit.setExemplarCheckOriginal(false);
                return proximitySplit;
            }
        },
        PS_R1_MQUICK() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R1.configure(proximitySplit);
                proximitySplit.setReduceSplitTestSize(true);
                proximitySplit.setEarlyAbandonDistances(true);
                proximitySplit.setExemplarCheckOriginal(false);
                return proximitySplit;
            }
        },
        PS_R10_MQUICK() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R10.configure(proximitySplit);
                proximitySplit.setReduceSplitTestSize(true);
                proximitySplit.setEarlyAbandonDistances(true);
                proximitySplit.setExemplarCheckOriginal(false);
                return proximitySplit;
            }
        },
        PS_R5() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R1.configure(proximitySplit);
                proximitySplit.setR(5);
                return proximitySplit;
            }
        },
        PS_R10() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R1.configure(proximitySplit);
                proximitySplit.setR(10);
                return proximitySplit;
            }
        },
        PS_R10_QUICK() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R10.configure(proximitySplit);
                proximitySplit.setEarlyAbandonDistances(true);
                proximitySplit.setExemplarCheckOriginal(false);
                return proximitySplit;
            }
        },
        PS_R5_QUICK() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R5.configure(proximitySplit);
                proximitySplit.setEarlyAbandonDistances(true);
                proximitySplit.setExemplarCheckOriginal(false);
                return proximitySplit;
            }
        },
        PS_R1_QUICK() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R1.configure(proximitySplit);
                proximitySplit.setEarlyAbandonDistances(true);
                proximitySplit.setExemplarCheckOriginal(false);
                return proximitySplit;
            }
        },
        PS_RR5() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R5.configure(proximitySplit);
                proximitySplit.setRandomR(true);
                return proximitySplit;
            }
        },
        PS_RR10() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R10.configure(proximitySplit);
                proximitySplit.setRandomR(true);
                return proximitySplit;
            }
        },
        PS_R1_I() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R1.configure(proximitySplit);
                proximitySplit.setRandomIntervals(true);
                proximitySplit.setMinIntervalSize(3);
                return proximitySplit;
            }
        },
        PS_R5_I() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R5.configure(proximitySplit);
                proximitySplit.setRandomIntervals(true);
                proximitySplit.setMinIntervalSize(3);
                return proximitySplit;
            }
        },
        PS_R10_I() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R10.configure(proximitySplit);
                proximitySplit.setRandomIntervals(true);
                proximitySplit.setMinIntervalSize(3);
                return proximitySplit;
            }
        },
        PS_R20_I() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_R10.configure(proximitySplit);
                proximitySplit.setRandomIntervals(true);
                proximitySplit.setMinIntervalSize(3);
                proximitySplit.setR(20);
                return proximitySplit;
            }
        },
        PS_RR5_I() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_RR5.configure(proximitySplit);
                proximitySplit.setRandomIntervals(true);
                proximitySplit.setMinIntervalSize(3);
                return proximitySplit;
            }
        },
        PS_RR10_I() {
            @Override
            public <B extends ProximitySplit> B configureFromEnum(B proximitySplit) {
                proximitySplit = PS_RR10.configure(proximitySplit);
                proximitySplit.setRandomIntervals(true);
                proximitySplit.setMinIntervalSize(3);
                return proximitySplit;
            }
        }
    }

    public ProximitySplit() {
        Config.PS_R5.configure(this);
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
    private Instances trainData;
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
    // whether to use intervals
    private boolean randomIntervals;
    // the min interval size if using intervals
    private int minIntervalSize;
    // optional interval transformer
    private IntervalTransform intervalTransform;
    // the intervaled train data
    private Instances modifiedTrainData;
    // whether to reduce the number of instances used in testing split quality
    private boolean reduceSplitTestSize;

    /**
     * build the split using the data provided
     */
    public void buildClassifier() throws Exception {
        super.buildClassifier(trainData);
        // make a map to store the best X splits
        final PrunedMultimap<Double, SplitCandidate> map = PrunedMultimap.desc();
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
        // backup the train data as modifications may be made during splitting (e.g. if doing intervals)
        final Instances origTrainData = new Instances(trainData);
        modifiedTrainData = null;
        // reduce the test set for split quality
        Instances test = trainData;
        Instances leftOverTest = null;
        if(reduceSplitTestSize) {
            test = new Instances(trainData, 0);
            leftOverTest = new Instances(trainData, 0);
            // change the view of the data into per class
            final Map<Double, Instances> instancesByClass = Utilities.instancesByClass(this.trainData);
            for(Map.Entry<Double, Instances> entry : instancesByClass.entrySet()) {
                final Instances instances = entry.getValue();
                final int size = Math.min(instances.size(), (int) Math.ceil(Math.sqrt(instances.size() * 10)));
                final Set<Integer> indices = new HashSet<>(RandomUtils.choiceIndex(instances.size(), rand, size));
                for(int i = 0; i < instances.size(); i++) {
                    final Instance instance = instances.get(i);
                    if(indices.contains(i)) {
                        test.add(instance);
                    } else {
                        leftOverTest.add(instance);
                    }
                }
            }
        }
        // for every split attempt
        for(int i = 0; i < r; i++) {
            // pick the distance function
            pickDistanceFunction();
            // pick the exemplars
            pickExemplars();
            // setup the partitions
            List<Instances> partitions = Lists.newArrayList(exemplarGroups.size());
            for(List<Instance> group : exemplarGroups) {
                partitions.add(new Instances(this.trainData, 0));
            }
            distanceFunction.setInstances(this.trainData);
            // go through every instance and find which partition it should go into. This should be the partition
            // with the closest exemplar associate
            for(final Instance instance : test) {
                final int index = getPartitionIndexFor(instance);
                final Instances closestPartition = partitions.get(index);
                closestPartition.add(instance);
            }
            // find the score of this split attempt, i.e. how good it is
            double score = partitionScorer.findScore(test, partitions);
            // chuck into a container to keep for later
            SplitCandidate splitCandidate = new SplitCandidate(exemplarGroups, distanceFunction, partitions, score, intervalTransform);
            // add it to the map. The map will handle whether the split attempt was any good and should be kept
            map.put(score, splitCandidate);
        }
        // choose the best of the R splits. The map handles the tie break if necessary
        SplitCandidate choice = RandomUtils.choice(new ArrayList<>(map.values()), rand);
        // populate the fields of this split from the split attempt
        setDistanceFunction(choice.getDistanceFunction());
        setIntervalTransform(choice.getIntervalTransform());
        setExemplarGroups(choice.getExemplars());
        setPartitions(choice.getPartitions());
        setScore(choice.getScore());
        // reduced the split size so need to test all train instances
        if(reduceSplitTestSize) {
            // go through every instance and find which partition it should go into. This should be the partition
            // with the closest exemplar associate
            for(final Instance instance : leftOverTest) {
                final int index = getPartitionIndexFor(instance);
                final Instances closestPartition = partitions.get(index);
                closestPartition.add(instance);
            }
//            // find the score of this split attempt, i.e. how good it is
//            double score = partitionScorer.findScore(this.trainData, partitions);
//            setScore(score);
        }
        // move any modified version of the train data elsewhere so the trainData is set back to the original
        modifiedTrainData = trainData;
        trainData = origTrainData;
        ResultUtils.setInfo(trainResults, this, trainData);
    }

    public boolean isReduceSplitTestSize() {
        return reduceSplitTestSize;
    }

    public void setReduceSplitTestSize(final boolean reduceSplitTestSize) {
        this.reduceSplitTestSize = reduceSplitTestSize;
    }

    // small helper class to contain a split. This is used to temporarily hold split results while
    // comparing R splits to pick the best
    private static class SplitCandidate {

        SplitCandidate(final List<List<Instance>> exemplars,
                         final DistanceFunction distanceFunction,
                         final List<Instances> partitions, final double score,
                         final IntervalTransform intervalTransform) {
            this.exemplars = exemplars;
            this.distanceFunction = distanceFunction;
            this.partitions = partitions;
            this.score = score;
            this.intervalTransform = intervalTransform;
        }

        private final DistanceFunction distanceFunction;
        private final List<Instances> partitions;
        private final double score;
        private final List<List<Instance>> exemplars;
        private final IntervalTransform intervalTransform;

        public List<List<Instance>> getExemplars() {
            return exemplars;
        }

        public DistanceFunction getDistanceFunction() {
            return distanceFunction;
        }

        public List<Instances> getPartitions() {
            return partitions;
        }

        public double getScore() {
            return score;
        }

        public IntervalTransform getIntervalTransform() {
            return intervalTransform;
        }
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        setTrainData(trainData);
        buildClassifier();
    }

    /**
     * pick the distance function
     */
    private void pickDistanceFunction() {
        Instances dataForBuilding = trainData;
        if(randomIntervals) {
            Assert.assertTrue(minIntervalSize > 0);
            // suppose we're looking at instances of length 41.
            // the last value is the class label, therefore there's a ts of 40.
            // the max length of an interval is therefore numAttributes() - 1. +1 for random call is cancelled out
            // by the -1 for num attributes including the class label
            // if a min interval size is then included, say 3, then the max size of the interval should be 40 - 3 =
            // 37. The min size can be subtracted from the rand call and added after to ensure rand numbers between
            // min and max length (3 and 40).
            final int length = rand.nextInt(trainData.numAttributes() - minIntervalSize) + minIntervalSize;
            Assert.assertTrue(length > 0);
            // the start point is dependent on the length. Max length of 40 then the start can only be 0. Min length
            // of 3 then the start can be anywhere between 0..37 inclusively.
            // The start can therefore lie anywhere from 0 to tsLen - intervalLen inclusively. (shortest interval
            // would be 3, 40 - 3 = 37, checks out). +1 for random call is cancelled out by the -1 for num attributes
            // including the class label
            final int start = rand.nextInt(trainData.numAttributes() - length);
            final Interval interval = new Interval(start, length);
            setIntervalTransform(new IntervalTransform(interval));
            dataForBuilding = intervalTransform.transform(trainData);
        }
        // pick a random space
        distanceFunctionSpaceBuilder = RandomUtils.choice(distanceFunctionSpaceBuilders, rand);
        // built that space
        distanceFunctionSpace = distanceFunctionSpaceBuilder.build(dataForBuilding);
        // randomly pick the distance function / parameters from that space
        final ParamSet paramSet = RandomSearchIterator.choice(rand, distanceFunctionSpace);
        // there is only one distance function in the ParamSet returned
        DistanceFunction distanceFunction = (DistanceFunction) paramSet.getSingle(DistanceMeasure.DISTANCE_MEASURE_FLAG);
        if(randomIntervals) {
            if(distanceFunction instanceof TransformDistanceMeasure) {
                final TransformDistanceMeasure tdf = (TransformDistanceMeasure) distanceFunction;
                tdf.setTransformer(new TransformPipeline(Lists.newArrayList(tdf.getTransformer(), intervalTransform)));
                if(tdf.isAltTransformer()) {
                    tdf.setAltTransformer(new TransformPipeline(Lists.newArrayList(tdf.getAltTransformer(), intervalTransform)));
                }
            } else {
                distanceFunction = new BaseTransformDistanceMeasure(DistanceMeasure.getName(distanceFunction), intervalTransform, distanceFunction);
            }
        }
        setDistanceFunction(distanceFunction);
    }

    /**
     * pick exemplars from the given dataset
     */
    private void pickExemplars() {
        // change the view of the data into per class
        final Map<Double, Instances> instancesByClass = Utilities.instancesByClass(this.trainData);
        // pick one exemplar per class
        List<List<Instance>> exemplarGroups = Lists.newArrayList(instancesByClass.size());
        for(Double classLabel : instancesByClass.keySet()) {
            final Instances instanceClass = instancesByClass.get(classLabel);
            final Instance exemplar = RandomUtils.choice(instanceClass, rand);
            // orig pf always calls for a random number even if the instances has only one instance. The choice
            // function doesn't source a random number given a list of size 1, therefore the random number call must
            // be made here to match orig pf random number sequence
            if(instanceClass.size() == 1) {
                rand.nextInt(1);
            }
            exemplarGroups.add(Lists.newArrayList(exemplar));
        }
        setExemplarGroups(exemplarGroups);
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
        final Integer closestPartitionIndex = Utilities.randPickOne(closestPartitionIndices, rand);
        // no-op, but must sample the random source to match orig pf. The orig implementation always samples the
        // random irrelevant of whether the list size is 1 or more. We only sample IFF the list size is larger than 1
        // . Therefore we'll sample the random if the list is exactly equal to 1 in size.
        if(matchOriginalPFRandomCalls && numIndices == 1) {
            rand.nextInt(numIndices);
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

    public Instances getTrainData() {
        return trainData;
    }

    public void setTrainData(Instances trainData) {
        Assert.assertNotNull(trainData);
        this.trainData = trainData;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(getClass().getSimpleName() + "{" +
                             "score=" + score +
                             ", dataSize=" + trainData.size());
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

    public boolean isRandomIntervals() {
        return randomIntervals;
    }

    public void setRandomIntervals(final boolean randomIntervals) {
        this.randomIntervals = randomIntervals;
        if(randomIntervals && minIntervalSize <= 0) {
            setMinIntervalSize(3);
        }
    }

    public void setMaxR(final int maxR) {
        this.maxR = maxR;
    }

    public boolean isExemplarCheckOriginal() {
        return exemplarCheckOriginal;
    }

    public int getMinIntervalSize() {
        return minIntervalSize;
    }

    public void setMinIntervalSize(final int minIntervalSize) {
        this.minIntervalSize = minIntervalSize;
    }

    public IntervalTransform getIntervalTransform() {
        return intervalTransform;
    }

    public void setIntervalTransform(final IntervalTransform intervalTransform) {
        this.intervalTransform = intervalTransform;
    }

    public Instances getModifiedTrainData() {
        return modifiedTrainData;
    }
}
