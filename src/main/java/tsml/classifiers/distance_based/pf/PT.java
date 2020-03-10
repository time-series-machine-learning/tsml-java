package tsml.classifiers.distance_based.pf;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.GcMemoryWatchable;
import tsml.classifiers.StopWatchTrainTimeable;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.distance_based.pf.partition.Partitioner;
import tsml.classifiers.distance_based.proximity.tree.BaseTreeNode;
import tsml.classifiers.distance_based.proximity.tree.BaseTree;
import tsml.classifiers.distance_based.proximity.tree.TreeNode;
import utilities.MemoryWatcher;
import utilities.StopWatch;
import utilities.iteration.RandomListIterator;
import weka.core.Instances;

import java.io.Serializable;
import java.util.ListIterator;
import java.util.function.Supplier;

// todo make rand sources use a single rand rather than a seed so we can swap the source easily - or we could just clone it instead in use cases
// todo built

public class PT extends EnhancedAbstractClassifier implements TrainTimeContractable, StopWatchTrainTimeable, GcMemoryWatchable {

    public PT() {

    }

    public PT(PT other) {
        throw new UnsupportedOperationException();
    }

    private boolean rebuild = true;
    private StopWatch trainTimer = new StopWatch();
    private MemoryWatcher memoryWatcher = new MemoryWatcher();
    private BaseTree<Split> tree;
    private Supplier<Partitioner> partitionerBuilder; // todo
    private Supplier<ListIterator<TreeNode<Split>>> nodeIteratorBuilder = () -> new RandomListIterator<>(rand);
    private ListIterator<TreeNode<Split>> nodeIterator;
    private Scorer scorer = Scorer.giniScore;
    private StoppingCondition stoppingCondition = (node, pt) -> node.getElement().getScore() == 1;

    public interface StoppingCondition extends Serializable {
        boolean stop(TreeNode<Split> node, PT pt);
    }

    public static class Split {
        private Partitioner partitioner;
        private Instances data;
        private double score = -1;

        public Split(Partitioner partitioner, Instances data) {
            setData(data);
            setPartitioner(partitioner);
        }

        public Partitioner getPartitioner() {
            return partitioner;
        }

        public void setPartitioner(Partitioner partitioner) {
            this.partitioner = partitioner;
        }

        public Instances getData() {
            return data;
        }

        public void setData(Instances data) {
            this.data = data;
        }

        public double getScore() {
            return score;
        }

        public void setScore(double score) {
            this.score = score;
        }
    }

    @Override
    public void setRebuild(boolean rebuild) {
        this.rebuild = rebuild;
        super.setRebuild(rebuild);
    }

    private TreeNode<Split> addNode(Instances data) {
        Partitioner partitioner = partitionerBuilder.get();
        Split split = new Split(partitioner, data);
        TreeNode<Split> node = new BaseTreeNode<>(split);
        if(!stoppingCondition.stop(node, this)) {
            nodeIterator.add(node);
        }
        return node;
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        if(rebuild) {
            memoryWatcher.resetAndEnable();
            trainTimer.resetAndEnable();
        }
        super.buildClassifier(trainData);
        if(rebuild) {
            rebuild = false;
            tree = new BaseTree<>();
            nodeIterator = nodeIteratorBuilder.get();
            addNode(trainData);
        }
        trainTimer.lap();
        while (hasNext() && hasRemainingTrainTime()) {
            BaseTreeNode<Split> node = nodeIterator.next();
            Split split = node.getElement();
            Partitioner partitioner = split.getPartitioner();
            Instances data = split.getData();
            partitioner.setTrainTimeLimit(getRemainingTrainTimeNanos());
            partitioner.buildClassifier(data);
            double score = scorer.findScore(data, partitioner.getPartitions());
            split.setScore(score);
            for(Instances partition : partitioner.getPartitions()) {
                TreeNode<Split> subNode = addNode(partition);
                node.addChild(subNode);
            }
            partitioner.clean();
            trainTimer.lap();
        }
        trainTimer.disable();
        memoryWatcher.disable();
    }

    public boolean hasNext() {
        return nodeIterator.hasNext();
    }

    @Override
    public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    @Override
    public StopWatch getTrainTimer() {
        return trainTimer;
    }
}
