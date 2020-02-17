package tsml.classifiers.distance_based.pf.partition;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import tsml.classifiers.*;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import utilities.MemoryWatcher;
import utilities.Rand;
import utilities.StopWatch;
import utilities.Utilities;
import utilities.collections.PrunedMultimap;
import utilities.iteration.RandomListIterator;
import utilities.serialisation.SerConsumer;
import utilities.serialisation.SerFunction;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

// todo checkpointing
// todo scoring
// todo predict times

public class ExemplarPartitioner extends EnhancedAbstractClassifier implements Partitioner, Rand, TrainTimeContractable, StopWatchTrainTimeable, GcMemoryWatchable {

    public ExemplarPartitioner() {

    }

    public ExemplarPartitioner(ExemplarPartitioner other) {
        throw new UnsupportedOperationException();
    }

    private List<Instances> partitions;
    private DistanceFunction distanceFunction;
    private List<Instance> exemplars;
    private boolean earlyAbandon = true;
    private SerConsumer<Instances> setupFunction = (SerConsumer<Instances>) instances -> {};
    private MemoryWatcher memoryWatcher = new MemoryWatcher();
    private StopWatch trainTimer = new StopWatch();
    private SerFunction<Instances, Iterator<Instance>> iteratorBuilder = (SerFunction<Instances, Iterator<Instance>>) instances -> new RandomListIterator<>(instances, rand.nextInt());
    private Iterator<Instance> iterator;
    private Map<Instance, Instances> splitByExemplarMap;
    private boolean cleanAfterBuild = false;
    private boolean rebuild = true;

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance closestExemplar = findClosestExemplar(instance);
        double[] distribution = new double[numClasses];
        distribution[(int) closestExemplar.classValue()]++;
        return distribution;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return Utilities.argMax(distributionForInstance(instance), rand);
    }

    public void buildClassifier(Instances trainData) throws Exception {
        if(trainData == null) {
            throw new IllegalStateException("trainData cannot be null");
        }
        if(distanceFunction == null) {
            throw new IllegalStateException("distance function cannot be null");
        }
        if(exemplars == null) {
            throw new IllegalStateException("exemplars cannot be null");
        }
        if(rebuild) {
            memoryWatcher.resetAndEnable();
            trainTimer.resetAndEnable();
        }
        super.buildClassifier(trainData);
        if(rebuild) {
            setupFunction.accept(trainData);
            rebuild = false;
            partitions = new ArrayList<>();
            splitByExemplarMap = new HashMap<>();
            for(Instance exemplar : exemplars) {
                Instances part = new Instances(trainData, 0);
                splitByExemplarMap.put(exemplar, part);
                partitions.add(part);
            }
            iterator = iteratorBuilder.apply(trainData);
        }
        trainTimer.lap();
        while (hasNext() && hasRemainingTrainTime()) {
            Instance instance = iterator.next();
            Instance closestExemplar = findClosestExemplar(instance);
            Instances instances = splitByExemplarMap.get(closestExemplar);
            instances.add(instance);
            trainTimer.lap();
        }
        if(!hasNext()) {
            if(cleanAfterBuild) {
                clean();
            }
            splitByExemplarMap = null;
        }
        trainTimer.disable();
        memoryWatcher.disable();
    }

    @Override
    public void clean() {
        partitions = null;
    }

    public boolean hasNext() {
        return iterator.hasNext();
    }

    public List<Double> findDistanceToExemplars(Instance instance) {
        List<Double> distances = new ArrayList<>(exemplars.size());
        double min = DistanceMeasure.MAX_DISTANCE;
        for(Instance exemplar : exemplars) {
            double distance = distanceFunction.distance(instance, exemplar, min);
            if(earlyAbandon) {
                min = Math.min(min, distance);
            }
        }
        return distances;
    }

    public Integer findClosestExemplarIndex(Instance instance) {
        PrunedMultimap<Double, Integer> map = PrunedMultimap.asc(ArrayList::new);
        map.setSoftLimit(1);
        map.setSeed(rand.nextInt());
        List<Double> distances = findDistanceToExemplars(instance);
        if(distances.size() != exemplars.size()) {
            throw new IllegalStateException("shouldn't happen");
        }
        for(int i = 0; i < exemplars.size(); i++) {
            map.put(distances.get(i), i);
        }
        return Utilities.randPickOne(map.values(), rand);
    }

    public Instance findClosestExemplar(Instance instance) {
        return exemplars.get(findClosestExemplarIndex(instance));
    }

    @Override
    public int getPartitionIndex(Instance instance) {
        return findClosestExemplarIndex(instance);
    }

    public SerConsumer<Instances> getSetupFunction() {
        return setupFunction;
    }

    public void setSetupFunction(SerConsumer<Instances> setupFunction) {
        this.setupFunction = setupFunction;
    }

    public SerFunction<Instances, Iterator<Instance>> getIteratorBuilder() {
        return iteratorBuilder;
    }

    public void setIteratorBuilder(SerFunction<Instances, Iterator<Instance>> iteratorBuilder) {
        this.iteratorBuilder = iteratorBuilder;
    }

    public boolean isCleanAfterBuild() {
        return cleanAfterBuild;
    }

    public void setCleanAfterBuild(boolean cleanAfterBuild) {
        this.cleanAfterBuild = cleanAfterBuild;
    }

    public boolean isEarlyAbandon() {
        return earlyAbandon;
    }

    public void setEarlyAbandon(boolean earlyAbandon) {
        this.earlyAbandon = earlyAbandon;
    }

    @Override
    public List<Instances> getPartitions() {
        return partitions;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public void setDistanceFunction(DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }

    public List<Instance> getExemplars() {
        return exemplars;
    }

    public void setExemplars(List<Instance> exemplars) {
        this.exemplars = exemplars;
    }

    @Override
    public void setRandom(Random random) {
        this.rand = random;
    }

    @Override
    public Random getRandom() {
        return rand;
    }

    @Override
    public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    @Override
    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override
    public StopWatch getTrainEstimateTimer() {
        return new StopWatch();
    }

    @Override
    public void setRebuild(boolean rebuild) {
        this.rebuild = rebuild;
        super.setRebuild(rebuild);
    }
}
