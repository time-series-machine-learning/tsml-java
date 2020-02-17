package tsml.classifiers.distance_based.pf;

import tsml.classifiers.*;
import tsml.classifiers.distance_based.pf.partition.Partitioner;
import tsml.classifiers.distance_based.pf.tree.Node;
import tsml.classifiers.distance_based.pf.tree.Tree;
import utilities.*;
import utilities.serialisation.SerConsumer;
import utilities.serialisation.SerFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Function;

public class ProximityTree extends EnhancedAbstractClassifier implements Rand, StopWatchTrainTimeable, TrainTimeContractable, GcMemoryWatchable {
    // todo estimate train const / actually do it

    private Tree<Partitioner> tree;
    private boolean rebuild = true;
    private SerConsumer<Instances> trainSetupFunction = instances -> {};
    private List<Node<Partitioner>> backlog;
    private SerFunction<Instances, Partitioner> partitionerBuilder;
    private long trainTimeLimitNanos = -1;
    private StopWatch trainTimer = new StopWatch();
    private MemoryWatcher memoryWatcher = new MemoryWatcher();

    @Override
    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override
    public StopWatch getTrainEstimateTimer() {
        return new StopWatch();
    }

    @Override
    public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    @Override
    public void setTrainTimeLimitNanos(long trainTimeLimitNanos) {
        this.trainTimeLimitNanos = trainTimeLimitNanos;
    }

    @Override
    public long getTrainTimeLimitNanos() {
        return trainTimeLimitNanos;
    }

    @Override public void setRebuild(final boolean rebuild) {
        this.rebuild = rebuild;
        super.setRebuild(rebuild);
    }

    private Partitioner buildPartitioner(Instances data) {
        return partitionerBuilder.apply(data);
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        if(rebuild) {
            trainTimer.resetAndEnable();
        }
        super.buildClassifier(trainData);
        if(rebuild) {
            rebuild = false;
            trainSetupFunction.accept(trainData);
            tree = new Tree<>();
            backlog = new ArrayList<>();
            Partitioner partitioner = buildPartitioner(trainData);
            partitioner.buildClassifier(trainData);
            Node<Partitioner> node = new Node<>(partitioner);
            backlog.add(node);
            tree.setRoot(node);
        }
        while (hasNext()) {
            next();
        }
        trainTimer.disable();
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Node<? extends Partitioner> node = tree.getRoot();
        boolean stop = false;
        double[] distribution = new double[numClasses];
        while (!stop) {
            Partitioner partitioner = node.getElement();
            int index = partitioner.getPartitionIndex(instance);
            if(node.isLeaf()) {
                node = null;
                stop = true;
                distribution[index]++;
            } else {
                node = node.getChildren().get(index);
            }
        }
        return distribution;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return Utilities.argMax(distributionForInstance(instance), rand);
    }

    @Override
    public long predictNextTrainTimeNanos() {

    }

    public boolean hasNext() {
        return !backlog.isEmpty() && hasRemainingTraining();
    }

    public ProximityTree next() {
        Node<Partitioner> node = backlog.remove(0);
        Partitioner partitioner = node.getElement();
        partitioner.buildClassifier();
        List<Instances> parts = partitioner.getPartitions();
        for(Instances part : parts) {
            Partitioner sub = buildPartitioner(part);
            sub.setData(part);
            Node<Partitioner> child = new Node<>(sub);
            node.getChildren().add(child);
            backlog.add(child);
            partitioner.cleanUp();
        }
        return this;
    }

    public SerConsumer<Instances> getTrainSetupFunction() {
        return trainSetupFunction;
    }

    public void setTrainSetupFunction(Consumer<Instances> trainSetupFunction) {
        this.trainSetupFunction = trainSetupFunction::accept;
    }

    public Function<Instances, Partitioner> getPartitionerBuilder() {
        return partitionerBuilder;
    }

    public void setPartitionerBuilder(Function<Instances, Partitioner> partitionerBuilder) {
        this.partitionerBuilder = partitionerBuilder::apply;
    }

    public static void main(String[] args) throws Exception {
        ProximityTree pt = ProximityTreeConfigs.buildDefaultProximityTree();
        ClassifierTools.trainAndTest("/bench/datasets", "GunPoint", 0, pt);
    }

    @Override
    public void setRandom(Random random) {
        this.rand = random;
    }

    @Override
    public Random getRandom() {
        return rand;
    }
}
